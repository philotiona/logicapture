"""Reproduce the assignment measurements and export the same report as the app."""
from dataclasses import asdict, replace
from io import BytesIO
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from src.interface.panels import group_board
from .dataset import (demo_dataset, read_dataset_zip, evaluation_cohorts,
                      dataset_fingerprint, pixel_digest)
from .inference import (ROOT, MODEL_PATH, Settings, load_model, evaluate_images, metrics,
                        parameter_sweeps, transform_image)
from .report import MODEL_COMPARISON, build_export
from .provenance import validate_training_record


def run_experiment(samples, adapted_path, training, output, settings=Settings()):
    """Measure independent cohorts; keep the personal-photo requirement explicit."""
    samples = list(samples)
    splits = validate_training_record(samples, training, adapted_path)
    fingerprint = dataset_fingerprint(samples)
    initial, final = evaluation_cohorts(splits["test"])
    if len(initial) < 5 or len(final) < 5:
        raise ValueError("This assignment experiment needs ten unique held-out source images.")
    output = Path(output)
    targets = [output/name for name in ("experiment.zip", "REPORT.md", "analysis.json")]
    if any(path.exists() for path in targets):
        raise ValueError("Choose an output directory without an existing experiment report.")
    before, after = load_model(MODEL_PATH), load_model(adapted_path)

    def compare(group):
        rows = {}
        for key, model in (("before", before), ("after", after)):
            values = evaluate_images([s.image for s in group], [s.label for s in group], model, settings)
            rows[key] = [{**row, "image_id": pixel_digest(s.image), "image_name": s.name}
                         for row, s in zip(values, group)]
            rows["metrics_"+key] = metrics(rows[key])
        return rows

    initial_result, final_result = compare(initial), compare(final)
    selected = next((sample for sample in initial if sample.label == "quadrilateral"), initial[0])
    sweeps = parameter_sweeps(selected.image, before, settings)
    for records in sweeps.values():
        for row in records:
            row["truth"] = selected.label
            row["correct"] = row["accepted"] and row["prediction"] == selected.label
    payload = {"metadata": {"source": "reproducible command-line experiment",
               "dataset_kind": "synthetic" if all(s.name.startswith("synthetic/") for s in samples) else "personal_photos",
               "settings": asdict(settings), "dataset_fingerprint": fingerprint,
               "split_seed": training["seed"], "preprocessing_id": training["preprocessing_id"],
               "base_model_sha256": training["base_model_sha256"],
               "adapted_model_sha256": training["adapted_model_sha256"],
               "split_counts": {key: len(value) for key, value in splits.items()},
               "test_image": selected.name, "test_image_id": pixel_digest(selected.image),
               "sweep_model": "Pretrained CNN", "cnn_input": [48, 48],
               "sliders": ["confidence", "brightness", "contrast"]},
               "model_comparison": MODEL_COMPARISON, "sweeps": sweeps, "training": training,
               "initial": initial_result, "final": final_result}
    illustrations = {"selected_original.png": selected.image,
                     "selected_processed.png": transform_image(selected.image, settings)}
    for variant, options in (("original", settings), ("rotate20", replace(settings, rotation=20)),
                             ("hflip", replace(settings, flip=True))):
        illustrations[f"augmentation/{variant}.png"] = transform_image(selected.image, options)
    for index, sample in enumerate(initial[:3], start=1):
        for state in ("before", "after"):
            row = initial_result[state][index-1]
            illustrations[f"before_after/{index}_{state}.png"] = group_board(
                [transform_image(sample.image, settings)],
                [f"{state.title()}: {row['prediction']} / {row['confidence']:.1%}"], columns=1, tile=280)
    for index, sample in enumerate(final, start=1):
        illustrations[f"final/{index}.png"] = group_board([transform_image(sample.image, settings)],
            [f"Truth: {sample.label}"], columns=1, tile=200)
    data = build_export(payload, illustrations)
    output.mkdir(parents=True, exist_ok=True)
    targets[0].write_bytes(data)
    with ZipFile(BytesIO(data)) as archive:
        for path in targets[1:]:
            path.write_bytes(archive.read(path.name))
        for name in archive.namelist():
            if name.startswith("illustrations/"):
                path = output/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--dataset", type=Path)
    parser.add_argument("--adapted", type=Path, default=ROOT/"data/rgr/demo/model.keras")
    parser.add_argument("--training", type=Path, help="Defaults to training.json beside the adapted model")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/rgr-experiment")
    args = parser.parse_args()
    try:
        samples = demo_dataset(30, 2026) if args.demo else read_dataset_zip(args.dataset.read_bytes())
        record = args.training or args.adapted.with_name("training.json")
        training = json.loads(record.read_text(encoding="utf-8"))
        result = run_experiment(samples, args.adapted, training, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"initial": result["initial"]["metrics_after"],
                      "final": result["final"]["metrics_after"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
