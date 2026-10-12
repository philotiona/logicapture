"""Object-region neural perception; probability-preserving hard-symbolic bridge.

TensorFlow is imported only for explicit neural use. The OpenCV-only path and
Akinator remain independent. Future soft evaluators can consume the retained
distributions without rerunning perception or replacing the scene contract.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
from random import Random
import numpy as np
from PIL import Image

from src.scene import Scene, VisualObject
from src.scene.renderer import draw_scene

SHAPE_CLASSES = ("circle", "triangle", "quadrilateral", "polygon")
FILL_CLASSES = ("outline", "filled")


def normalize_region(region, size=48):
    image = Image.fromarray(region).convert("L") if not isinstance(region, Image.Image) else region.convert("L")
    image.thumbnail((size-8, size-8), Image.Resampling.BILINEAR)
    padded = Image.new("L", (size, size), "white")
    padded.paste(image, ((size-image.width)//2, (size-image.height)//2))
    return np.asarray(padded, dtype=np.float32)[..., None]/255


def build_attribute_model(seed=71):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.layers.Input((48, 48, 1))
    x = tf.keras.layers.Conv2D(12, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(24, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    shape = tf.keras.layers.Dense(len(SHAPE_CLASSES), activation="softmax", name="shape")(x)
    fill = tf.keras.layers.Dense(len(FILL_CLASSES), activation="softmax", name="fill")(x)
    return tf.keras.Model(inputs, {"shape": shape, "fill": fill}, name="object_attribute_cnn")


def predict_attributes(model, regions):
    if not regions: return []
    outputs = model(np.stack([normalize_region(r) for r in regions]), training=False)
    shapes, fills = np.asarray(outputs["shape"]), np.asarray(outputs["fill"])
    if shapes.shape != (len(regions), len(SHAPE_CLASSES)) or fills.shape != (len(regions), len(FILL_CLASSES)):
        raise ValueError("Attribute model has incompatible output heads")
    result = []
    for shape, fill in zip(shapes, fills):
        result.append((dict(zip(SHAPE_CLASSES, map(float, shape))), dict(zip(FILL_CLASSES, map(float, fill)))))
    return result


def apply_attribute_predictions(scene, regions, model, ids, geometric_guard=False):
    distributions = predict_attributes(model, [regions[i] for i in ids])
    for index, (shape, fill) in zip(ids, distributions):
        hard_shape = max(shape, key=shape.get)
        hard_fill = max(fill, key=fill.get) == "filled"
        confidence = min(max(shape.values()), max(fill.values()))
        if geometric_guard and (scene.objects[index].kind != "region" or
                                hard_shape != scene.objects[index].shape or
                                hard_fill != scene.objects[index].filled):
            # A closed-set softmax can be very certain on an unseen stroke.
            # Retain its output for research without overwriting contradictory
            # boundary/closure measurements in the default representation.
            scene.objects[index] = replace(scene.objects[index], shape_probabilities=shape,
                fill_probabilities=fill, attribute_source="neural_rejected")
            scene.warnings.append(f"Object {scene.objects[index].id}: CNN prediction conflicts with boundary evidence; measured attributes retained.")
            continue
        scene.objects[index] = replace(scene.objects[index], shape=hard_shape, filled=hard_fill,
                                       shape_probabilities=shape, fill_probabilities=fill,
                                       attribute_source="neural_argmax", confidence=confidence)
        if confidence < .6:
            scene.warnings.append(f"Object {scene.objects[index].id}: neural attributes have low probability ({confidence:.2f}); review the argmax decision.")
    return scene


def load_attribute_model(path="data/models/attributes.keras"):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def _isolated_dataset(per_class, seed):
    rng, noise = Random(seed), np.random.default_rng(seed)
    images, shape_labels, fill_labels = [], [], []
    for shape_index, shape in enumerate(SHAPE_CLASSES):
        for filled in (False, True):
            for _ in range(per_class):
                obj = VisualObject(0, shape, filled, area=rng.uniform(.03, .12), angle=rng.uniform(-180, 180))
                image = draw_scene(Scene([obj]), size=180, stroke=rng.randint(2, 5)).convert("L")
                array = np.array(image)
                # Match segmentation preprocessing: binary ink and a padded crop.
                mask = array < 160
                ys, xs = np.where(mask)
                pad = max(3, int(.08*max(xs.max()-xs.min(), ys.max()-ys.min())))
                array = np.where(mask, 0, 255).astype(np.uint8)
                array = array[max(0, ys.min()-pad):min(180, ys.max()+pad+1), max(0, xs.min()-pad):min(180, xs.max()+pad+1)]
                array = np.uint8(np.clip(array.astype(float)+noise.normal(0, rng.uniform(0, 9), array.shape), 0, 255))
                images.append(normalize_region(array))
                shape_labels.append(shape_index)
                fill_labels.append(int(filled))
    order = noise.permutation(len(images))
    return np.array(images)[order], np.array(shape_labels, np.int32)[order], np.array(fill_labels, np.int32)[order]


def train_attribute_model(output="data/models", epochs=12, per_class=90, seed=71):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = build_attribute_model(seed)
    optimizer = tf.keras.optimizers.Adam(.001)
    x, shapes, fills = _isolated_dataset(per_class, seed)
    test_x, test_shapes, test_fills = _isolated_dataset(max(20, per_class//4), seed+1000)

    @tf.function
    def step(batch, shape_labels, fill_labels):
        with tf.GradientTape() as tape:
            probabilities = model(batch, training=True)
            shape_loss = tf.keras.losses.sparse_categorical_crossentropy(shape_labels, probabilities["shape"])
            fill_loss = tf.keras.losses.sparse_categorical_crossentropy(fill_labels, probabilities["fill"])
            loss = tf.reduce_mean(shape_loss+fill_loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        indices = rng.permutation(len(x))
        epoch_losses = []
        for start in range(0, len(x), 32):
            batch = indices[start:start+32]
            epoch_losses.append(float(step(x[batch], shapes[batch], fills[batch])))
        losses.append(float(np.mean(epoch_losses)))
        print(f"Attribute CNN: epoch {epoch+1}/{epochs}; loss {losses[-1]:.4f}", flush=True)
    predicted = model(test_x, training=False)
    report = {"seed": seed, "epochs": epochs, "train_objects": len(x), "test_objects": len(test_x),
              "shape_classes": SHAPE_CLASSES, "fill_classes": FILL_CLASSES,
              "shape_accuracy": float(np.mean(np.argmax(predicted["shape"], axis=1) == test_shapes)),
              "fill_accuracy": float(np.mean(np.argmax(predicted["fill"], axis=1) == test_fills)),
              "losses": losses, "parameters": model.count_params(),
              "limitations": ["Independent synthetic isolated-object test only; not real handwriting.",
                              "Softmax scores are uncalibrated probabilities.",
                              "Segmentation errors remain an OpenCV limitation.",
                              "The polygon training class contains regular pentagons, not arbitrary topology."]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output/"attributes.keras")
    (output/"training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--per-class", type=int, default=90)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    if args.epochs < 1 or args.per_class < 1: parser.error("Training counts must be positive")
    train_attribute_model(args.output, args.epochs, args.per_class, args.seed)
