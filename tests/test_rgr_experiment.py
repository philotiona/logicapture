from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pytest

from src.rgr import experiment
from src.rgr.dataset import demo_dataset, dataset_fingerprint, split_samples, pixel_digest
from src.rgr.provenance import (PREPROCESSING_ID, file_sha256, split_manifest,
                                validate_training_record)


class Model:
    def __call__(self, batch, training=False):
        return {"shape": np.tile([.1, .1, .7, .1], (len(batch), 1))}


def _record(tmp_path, monkeypatch, samples, seed=2026):
    from src.rgr import inference
    base, adapted = tmp_path/"base.keras", tmp_path/"adapted.keras"
    base.write_bytes(b"independently bound base checkpoint")
    adapted.write_bytes(b"independently bound adapted checkpoint")
    monkeypatch.setattr(inference, "MODEL_PATH", base)
    monkeypatch.setattr(experiment, "MODEL_PATH", base)
    splits = split_samples(samples, seed)
    return adapted, {"dataset_fingerprint": dataset_fingerprint(samples), "seed": seed,
                     "split": split_manifest(splits),
                     "split_counts": {name: len(group) for name, group in splits.items()},
                     "preprocessing_id": PREPROCESSING_ID,
                     "base_model_sha256": file_sha256(base),
                     "adapted_model_sha256": file_sha256(adapted)}


def test_reproducible_experiment_exports_same_image_pairs_and_disjoint_final_set(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "load_model", lambda path: Model())
    samples = demo_dataset(15)
    adapted, record = _record(tmp_path, monkeypatch, samples)
    payload = experiment.run_experiment(samples, adapted, record, tmp_path/"report")
    initial, final = payload["initial"], payload["final"]
    initial_ids = [row["image_id"] for row in initial["before"]]
    assert initial_ids == [row["image_id"] for row in initial["after"]]
    assert len(initial_ids) == 5 and set(initial_ids).isdisjoint(row["image_id"] for row in final["before"])
    assert payload["metadata"]["dataset_kind"] == "synthetic"
    assert {row["value"] for row in payload["sweeps"]["threshold"]} == {.1, .3, .5, .7, .9}
    with ZipFile(BytesIO((tmp_path/"report/experiment.zip").read_bytes())) as archive:
        assert "illustrations/before_after/3_after.png" in archive.namelist()
        assert "illustrations/augmentation/rotate20.png" in archive.namelist()
        assert json.loads(archive.read("analysis.json"))["initial"] == initial
    assert (tmp_path/"report/illustrations/augmentation/rotate20.png").is_file()
    with pytest.raises(ValueError, match="existing experiment"):
        experiment.run_experiment(samples, adapted, record, tmp_path/"report")


def test_experiment_rejects_mismatched_training_sources_and_too_few_queries(tmp_path, monkeypatch):
    samples = demo_dataset(5)
    with pytest.raises(ValueError, match="different source"):
        experiment.run_experiment(samples, "adapted.keras", {}, tmp_path)
    adapted, record = _record(tmp_path, monkeypatch, samples)
    with pytest.raises(ValueError, match="ten unique"):
        experiment.run_experiment(samples, adapted, record, tmp_path/"report")


def test_nondefault_training_seed_keeps_both_experiment_cohorts_unseen(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "load_model", lambda path: Model())
    samples = demo_dataset(30)
    adapted, record = _record(tmp_path, monkeypatch, samples, seed=2027)
    result = experiment.run_experiment(samples, adapted, record, tmp_path/"seed2027")
    recorded = split_samples(samples, seed=2027)
    forbidden = {pixel_digest(s.image) for name in ("train", "validation") for s in recorded[name]}
    actual = {row["image_id"] for name in ("initial", "final") for row in result[name]["before"]}
    assert len(actual) == 10 and actual.isdisjoint(forbidden)
    assert actual <= {pixel_digest(s.image) for s in recorded["test"]}
    assert result["metadata"]["split_seed"] == 2027


@pytest.mark.parametrize("changed", ["seed", "manifest", "missing_hash", "adapted_bytes", "base_bytes", "preprocessing"])
def test_stale_or_unrelated_provenance_is_rejected_before_model_loading(tmp_path, monkeypatch, changed):
    samples = demo_dataset(15)
    adapted, record = _record(tmp_path, monkeypatch, samples)
    if changed == "seed":
        record["seed"] = 2027
    elif changed == "manifest":
        record["split"]["test"][0]["pixel_sha256"] = "0"*64
    elif changed == "missing_hash":
        record.pop("adapted_model_sha256")
    elif changed == "adapted_bytes":
        adapted.write_bytes(b"different valid model could occupy this path")
    elif changed == "base_bytes":
        experiment.MODEL_PATH.write_bytes(b"a newly trained base")
    elif changed == "preprocessing":
        record["preprocessing_id"] = "old-normalization"
    def forbidden_load(path):
        raise AssertionError("Stale provenance must fail before loading a model")
    monkeypatch.setattr(experiment, "load_model", forbidden_load)
    with pytest.raises(ValueError):
        experiment.run_experiment(samples, adapted, record, tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()


@pytest.mark.parametrize("seed", [None, True, -1, 2**32, "2026"])
def test_provenance_requires_original_valid_integer_seed(tmp_path, monkeypatch, seed):
    samples = demo_dataset(5)
    adapted, record = _record(tmp_path, monkeypatch, samples)
    record["seed"] = seed
    with pytest.raises(ValueError, match="integer split seed"):
        validate_training_record(samples, record, adapted)
