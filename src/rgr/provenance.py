"""Bind an adaptation record to its sources, split, and checkpoint bytes."""
from hashlib import sha256
from pathlib import Path

from .dataset import dataset_fingerprint, pixel_digest, split_samples

PREPROCESSING_ID = "rgr-letterbox-bilinear-linear-tones-normalize48-v1"


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest(splits):
    return {name: [{"name": sample.name, "label": sample.label,
                    "pixel_sha256": pixel_digest(sample.image)} for sample in group]
            for name, group in splits.items()}


def validate_training_record(samples, report, model_path):
    """Return the recorded seeded split, or reject stale/unrelated evidence.

    Dataset names, labels and canonical pixel identities must all match the
    complete saved manifest. A record without checkpoint hashes or a pipeline
    identity is legacy evidence and cannot establish a current adapted run.
    """
    samples = list(samples)
    if not isinstance(report, dict):
        raise ValueError("The training record must be a JSON object.")
    if report.get("dataset_fingerprint") != dataset_fingerprint(samples):
        raise ValueError("The adapted model's training record belongs to a different source dataset.")
    seed = report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("The training record needs its original integer split seed (0–4294967295).")
    splits = split_samples(samples, seed=seed)
    if report.get("split") != split_manifest(splits):
        raise ValueError("The saved training split manifest does not match the recorded seed and source images.")
    if report.get("split_counts") != {name: len(group) for name, group in splits.items()}:
        raise ValueError("The saved training split counts do not match the source manifest.")
    if report.get("preprocessing_id") != PREPROCESSING_ID:
        raise ValueError("The training record has a missing or incompatible preprocessing identity; retrain this dataset.")
    from .inference import MODEL_PATH
    for path, field, description in ((MODEL_PATH, "base_model_sha256", "pretrained base"),
                                     (model_path, "adapted_model_sha256", "adapted")):
        recorded_hash = report.get(field)
        if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
            raise ValueError(f"The training record is missing a valid {description} checkpoint hash; retrain this dataset.")
        try:
            actual_hash = file_sha256(path)
        except OSError as error:
            raise ValueError(f"Could not verify the {description} checkpoint: {error}") from error
        if actual_hash != recorded_hash:
            raise ValueError(f"The {description} checkpoint bytes do not match this training record.")
    return splits
