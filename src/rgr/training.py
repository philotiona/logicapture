"""Local head fine-tuning with source-isolated validation and untouched tests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np

from src.vision.attribute_cnn import SHAPE_CLASSES, load_attribute_model
from .dataset import dataset_fingerprint, demo_dataset, read_dataset_zip, split_samples
from .provenance import PREPROCESSING_ID, file_sha256, split_manifest

def _save_checkpoint(model, path):
    # This plain functional CNN has no custom assets. Package its public JSON
    # and weight exports in the native .keras layout. model.save() otherwise
    # creates an asset tempfile in the process-wide default Temp on Windows.
    # Explicit local staging avoids changing any global tempfile settings.
    import keras
    pending = path.with_name(".pending.keras")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as directory:
        weights = Path(directory)/"model.weights.h5"
        model.save_weights(weights)
        with ZipFile(pending, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("config.json", model.to_json())
            archive.writestr("metadata.json", json.dumps({"keras_version": keras.__version__,
                             "date_saved": datetime.now(timezone.utc).strftime("%Y-%m-%d@%H:%M:%S")}))
            archive.write(weights, "model.weights.h5")
    pending.replace(path)


def _probabilities(model, inputs):
    return np.asarray(model(inputs, training=False)["shape"])


def _metrics(probabilities, labels):
    predicted = probabilities.argmax(axis=1)
    correct = int(np.sum(predicted == labels))
    return {"accuracy": correct/len(labels), "correct": correct,
            "incorrect": len(labels)-correct, "count": len(labels),
            "mean_confidence": float(probabilities.max(axis=1).mean())}


def _augment(image, rng):
    """Training-only rotation and photometric augmentation after source split."""
    h, w = image.shape[:2]
    transform = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), float(rng.uniform(-10, 10)), 1.)
    array = cv2.warpAffine(image, transform, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    if rng.random() < .5:
        array = array[:, ::-1]
    return np.uint8(np.clip((array.astype(float)-127.5)*rng.uniform(.9, 1.1)+127.5+rng.uniform(-10, 10), 0, 255))


def fine_tune(samples, output_dir, epochs=4, seed=2026, progress=None) -> dict:
    """Load a fresh base, train its shape head, save best validation epoch.

    The base checkpoint is never modified. Output directories must be unused,
    preventing an accidental overwrite of a previous result or source dataset.
    """
    if not isinstance(epochs, int) or not 1 <= epochs <= 50:
        raise ValueError("Use 1–50 fine-tuning epochs.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Use an integer training seed from 0 to 4294967295.")
    samples = list(samples)
    splits = split_samples(samples, seed)
    fingerprint = dataset_fingerprint(samples)
    from .inference import MODEL_PATH, prepare_model_input
    base_path = Path(MODEL_PATH).resolve()
    if not base_path.is_file():
        raise ValueError("The pretrained attribute CNN is missing; train the base model first.")
    output = Path(output_dir).resolve()
    if (output == base_path.parent or output == base_path
            or output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Choose an empty output directory; existing models and datasets are preserved.")
    base_digest = file_sha256(base_path)
    import tensorflow as tf
    # Native Windows can otherwise allocate a large worker pool per session.
    # An already initialized UI runtime keeps its existing thread settings.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    tf.keras.utils.set_random_seed(seed)
    model = load_attribute_model(base_path)
    # Keep the feature extractor and fill head fixed. The RGR shape experiment
    # adapts only the existing shape head using independent labeled sources.
    for layer in model.layers:
        layer.trainable = layer.name == "shape"
    optimizer = tf.keras.optimizers.Adam(learning_rate=.002)
    labels = {name: np.asarray([SHAPE_CLASSES.index(s.label) for s in group], np.int32)
              for name, group in splits.items()}
    inputs = {name: np.stack([prepare_model_input(s.image) for s in group])
              for name, group in splits.items()}
    # Only validation selects the checkpoint. Test predictions are evaluated
    # after training and are never passed to the optimizer or epoch selector.
    rng = np.random.default_rng(seed)
    history, best_key, best_weights, selected_epoch = [], None, None, None

    @tf.function
    def train_step(batch, targets):
        with tf.GradientTape() as tape:
            probabilities = model(batch, training=True)["shape"]
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, probabilities))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        order = rng.permutation(len(splits["train"]))
        total_loss, total_seen = 0., 0
        for start in range(0, len(order), 32):
            indices = order[start:start+32]
            batch = np.stack([prepare_model_input(_augment(splits["train"][i].image, rng)) for i in indices])
            loss = float(train_step(batch, labels["train"][indices]))
            total_loss += loss*len(indices)
            total_seen += len(indices)
        validation = _probabilities(model, inputs["validation"])
        validation_loss = float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(labels["validation"], validation)))
        validation_accuracy = _metrics(validation, labels["validation"])["accuracy"]
        history.append({"epoch": epoch+1, "train_loss": total_loss/total_seen,
                        "validation_loss": validation_loss, "validation_accuracy": validation_accuracy})
        key = (validation_accuracy, -validation_loss)
        if best_key is None or key > best_key:
            best_key, best_weights, selected_epoch = key, model.get_weights(), epoch+1
        if progress:
            progress(f"Epoch {epoch+1}/{epochs}: validation accuracy {validation_accuracy:.1%}; loss {validation_loss:.4f}")
    model.set_weights(best_weights)
    after = _probabilities(model, inputs["test"])
    base = load_attribute_model(base_path)
    before = _probabilities(base, inputs["test"])
    report = {
        "schema_version": 1, "seed": seed, "epochs": epochs,
        "selected_epoch": selected_epoch, "checkpoint_selection": "validation accuracy, then validation loss",
        "classes": list(SHAPE_CLASSES), "dataset_fingerprint": fingerprint,
        "preprocessing_id": PREPROCESSING_ID,
        "source": "synthetic_demo" if all(s.name.startswith("synthetic/") for s in samples) else "uploaded_dataset",
        "base_model": str(base_path), "base_model_sha256": base_digest,
        "adaptation": "Only the shape head is fine-tuned; visual features and fill head are frozen.",
        "metrics_semantics": "Training-record accuracy uses unthresholded argmax predictions. Interactive comparison accuracy counts confidence-threshold rejections as incorrect.",
        "trainable_parameters": int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        "augmentation": ["training-only rotation ±10 degrees", "training-only horizontal flip", "training-only brightness and contrast"],
        "split": split_manifest(splits),
        "split_counts": {name: len(group) for name, group in splits.items()}, "history": history,
        "before": _metrics(before, labels["test"]), "after": _metrics(after, labels["test"]),
        "before_after": [{"name": sample.name, "true_label": sample.label,
                          "before_prediction": SHAPE_CLASSES[int(before[i].argmax())],
                          "before_confidence": float(before[i].max()),
                          "after_prediction": SHAPE_CLASSES[int(after[i].argmax())],
                          "after_confidence": float(after[i].max())} for i, sample in enumerate(splits["test"])],
        "limitations": ["Synthetic demo images are drawings, not real photographs; use personal photographs for the assignment.",
                        "Softmax probabilities are uncalibrated.",
                        "Head fine-tuning does not guarantee an improvement on unseen photographs.",
                        "Exact duplicate pixels are grouped; independently edited copies must be excluded before import."],
    }
    if file_sha256(base_path) != base_digest:
        raise RuntimeError("The base checkpoint changed during training; output was not saved.")
    output.mkdir(parents=True, exist_ok=True)
    model_path, report_path = output/"model.keras", output/"training.json"
    _save_checkpoint(model, model_path)
    report["adapted_model_sha256"] = file_sha256(model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "report_path": str(report_path), "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use explicitly synthetic sketches, not photographs")
    source.add_argument("--dataset", type=Path, help="ZIP with class/image.png files")
    parser.add_argument("--output", type=Path, default=Path("data/rgr"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        samples = demo_dataset(args.per_class, args.seed) if args.demo else read_dataset_zip(args.dataset.read_bytes())
        result = fine_tune(samples, args.output, args.epochs, args.seed, progress=print)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in result.items() if key != "report"}, indent=2))


if __name__ == "__main__":
    main()
