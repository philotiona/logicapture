from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import stat
from zipfile import ZipFile, ZipInfo

import numpy as np
from PIL import Image
import pytest

from src.rgr.dataset import (Sample, dataset_fingerprint, demo_dataset, pixel_digest,
                             read_dataset_zip, split_samples)


def _png(image):
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def _zip(entries):
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return stream.getvalue()


def test_dataset_zip_reads_only_labelled_images_in_memory():
    samples = demo_dataset(5)
    entries = [(f"dataset/{s.label}/{i}.png", _png(s.image)) for i, s in enumerate(samples)]
    entries += [("dataset/", b""), ("README.md", b"Personal isolated-shape photographs")]
    decoded = read_dataset_zip(_zip(entries))
    assert len(decoded) == 20
    assert {s.label for s in decoded} == {s.label for s in samples}
    assert all(np.array_equal(a.image, b.image) for a, b in zip(samples, decoded))
    splits = split_samples(decoded)
    assert {name: len(group) for name, group in splits.items()} == {"train": 12, "validation": 4, "test": 4}


@pytest.mark.parametrize("name", ["../circle/one.png", "/circle/one.png", "C:/circle/one.png",
                                  "unknown/one.png", "circle/file.exe"])
def test_dataset_import_rejects_unsafe_or_unlabelled_members(name):
    with pytest.raises(ValueError):
        read_dataset_zip(_zip([(name, _png(demo_dataset(5)[0].image))]))


def test_dataset_import_rejects_symlinks_and_decoded_pixel_budget(monkeypatch):
    raw = _zip([("circle/one.png", _png(demo_dataset(5)[0].image))])
    with pytest.raises(ValueError, match="unsafe path"):
        read_dataset_zip(raw.replace(b"circle/one.png", b"circle\\one.png"))
    link = ZipInfo("circle/link.png")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="symbolic link"):
        read_dataset_zip(_zip([(link, b"target.png")]))
    from src.rgr import dataset
    monkeypatch.setattr(dataset, "MAX_TOTAL_PIXELS", 20)
    with pytest.raises(ValueError, match="total limit"):
        read_dataset_zip(_zip([("circle/one.png", _png(np.zeros((10, 10, 3), np.uint8)))]))


def test_duplicate_sources_stay_together_and_conflicting_labels_fail():
    samples = demo_dataset(5)
    duplicate = Sample("second_copy.png", samples[0].image.copy(), samples[0].label)
    splits = split_samples(samples + [duplicate])
    owners = {}
    for name, group in splits.items():
        for sample in group:
            digest = pixel_digest(sample.image)
            assert digest not in owners or owners[digest] == name
            owners[digest] = name
    conflict = Sample("conflict.png", samples[0].image.copy(), "triangle")
    with pytest.raises(ValueError, match="conflicting"):
        split_samples(samples + [conflict])
    too_few = [s for s in samples if not s.name.endswith("0004.png")]
    with pytest.raises(ValueError, match="five unique"):
        split_samples(too_few)


def test_demo_and_split_are_deterministic_and_fingerprint_tracks_inputs():
    first, second = demo_dataset(5), demo_dataset(5)
    assert dataset_fingerprint(first) == dataset_fingerprint(list(reversed(second)))
    assert dataset_fingerprint(first) != dataset_fingerprint(demo_dataset(5, seed=2027))
    assert all(s.name.startswith("synthetic/") for s in first)
    assert {k: [s.name for s in v] for k, v in split_samples(first).items()} == {
        k: [s.name for s in v] for k, v in split_samples(list(reversed(second))).items()}


def test_real_fine_tuning_preserves_base_and_exports_independent_test_report(tmp_path):
    pytest.importorskip("tensorflow")
    from src.rgr.inference import MODEL_PATH, prepare_model_input
    from src.rgr.training import fine_tune
    from src.vision.attribute_cnn import load_attribute_model
    base_path = Path(MODEL_PATH)
    if not base_path.is_file():
        pytest.skip("Pretrained attribute CNN is unavailable")
    before_digest = sha256(base_path.read_bytes()).hexdigest()
    result = fine_tune(demo_dataset(5), tmp_path/"run", epochs=2, seed=2027)
    report = result["report"]
    from src.rgr.provenance import validate_training_record, file_sha256
    checked_splits = validate_training_record(demo_dataset(5), report, result["model_path"])
    assert len(checked_splits["test"]) == 4
    assert report["seed"] == 2027
    assert {row["name"] for row in report["before_after"]} == {
        sample.name for sample in split_samples(demo_dataset(5), seed=2027)["test"]}
    assert report["adapted_model_sha256"] == file_sha256(result["model_path"])
    assert report["source"] == "synthetic_demo"
    assert report["selected_epoch"] in (1, 2)
    assert report["before"]["count"] == report["after"]["count"] == 4
    assert len(report["before_after"]) == 4
    assert report["split_counts"] == {"train": 12, "validation": 4, "test": 4}
    owners = {}
    for name, group in report["split"].items():
        for source in group:
            assert source["pixel_sha256"] not in owners
            owners[source["pixel_sha256"]] = name
    assert json.loads(Path(result["report_path"]).read_text())["dataset_fingerprint"] == report["dataset_fingerprint"]
    assert sha256(base_path.read_bytes()).hexdigest() == before_digest
    base, adapted = load_attribute_model(base_path), load_attribute_model(result["model_path"])
    assert all(np.array_equal(a, b) for a, b in zip(base.layers[1].get_weights(), adapted.layers[1].get_weights()))
    assert any(not np.array_equal(a, b) for a, b in zip(base.get_layer("shape").get_weights(), adapted.get_layer("shape").get_weights()))
    original_sources = {sample.name: sample for sample in demo_dataset(5)}
    inputs = np.stack([prepare_model_input(original_sources[row["name"]].image) for row in report["before_after"]])
    restored = np.asarray(adapted(inputs, training=False)["shape"])
    assert np.allclose(restored.max(axis=1), [row["after_confidence"] for row in report["before_after"]])
    with pytest.raises(ValueError, match="empty output"):
        fine_tune(demo_dataset(5), tmp_path/"run", epochs=1)
