from hashlib import sha256
from io import BytesIO
import csv
import json
from pathlib import Path
import stat
from zipfile import ZipFile, ZipInfo

import numpy as np
from PIL import Image
import pytest

from src import image_tasks as tasks
from src.reasoning.features import FeatureBatch
from src.reasoning.general import GeneralVisualSolver

FIXTURES = Path(__file__).resolve().parents[1]/"data/fixtures/arrow_paths"


def encoded(image, format="PNG"):
    stream = BytesIO()
    Image.fromarray(image).save(stream, format=format)
    return stream.getvalue()


def archive(entries):
    stream = BytesIO()
    with ZipFile(stream, "w") as output:
        for name, data in entries:
            output.writestr(name, data)
    return stream.getvalue()


def drawing_task():
    def group(letter):
        return tuple(np.array(Image.open(FIXTURES/f"{letter}{i:02d}.png").convert("RGB")) for i in range(1, 5))
    return tasks.ImageTask("Arrow inspection", ("Opposing", "Same direction"), "drawings", group("A"), group("B"))


class PixelEncoder:
    encoder_id = "test-color-v1"

    def __init__(self):
        self.calls = []

    def encode(self, images, image_ids):
        self.calls.append(tuple(image_ids))
        values = np.stack([np.array(image).mean(axis=(0, 1))/255 for image in images])
        return FeatureBatch(tuple(image_ids), values, encoder_id=self.encoder_id)


def solid(rgb):
    return np.full((16, 16, 3), rgb, np.uint8)


def test_frozen_fit_reuses_only_reference_parameters_for_separate_batches():
    encoder = PixelEncoder()
    solver = GeneralVisualSolver(encoder)
    positive, negative = [solid((220, 20, 10)), solid((240, 30, 10))], [solid((10, 20, 220)), solid((10, 30, 240))]
    fitted = solver.fit(positive, negative)
    assert encoder.calls == [("A01", "A02", "B01", "B02")]
    coefficients = fitted.fitted.model.coefficients.copy()
    red, blue = solid((230, 25, 10)), solid((10, 25, 230))
    first = fitted.predict([red], image_ids=["new-red"])
    mixed = fitted.predict([blue, red], image_ids=["new-blue", "another-red"])
    assert first.predictions[0].label is True
    assert [p.label for p in mixed.predictions] == [False, True]
    assert first.predictions[0].score == pytest.approx(mixed.predictions[1].score)
    np.testing.assert_array_equal(coefficients, fitted.fitted.model.coefficients)
    assert fitted.diagnostics["query_labels_used"] is False
    assert fitted.predict([]).predictions == []
    with pytest.raises(ValueError, match="duplicates"):
        fitted.predict([positive[0]])
    with pytest.raises(ValueError, match="overlap"):
        fitted.predict([red], image_ids=["A01"])
    with pytest.raises(ValueError, match="unique"):
        fitted.predict([red, blue], image_ids=["query", "query"])


def test_task_round_trip_preserves_visible_pixels_names_and_fit_settings():
    task = drawing_task()
    task.groups = ("Қабылдау", "Проверить")
    loaded = tasks.load_task(task.export())
    assert task.fingerprint == loaded.fingerprint
    assert loaded.groups == task.groups and loaded.representation == task.representation
    assert all(np.array_equal(a, b) for a, b in zip(task.positive+task.negative, loaded.positive+loaded.negative))


@pytest.mark.parametrize("change", ["corrupt_hash", "version", "label", "extra", "repeat", "non_string_path"])
def test_task_import_rejects_changed_evidence_or_invalid_manifests(change):
    with ZipFile(BytesIO(drawing_task().export())) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(entries["task.json"])
    if change == "corrupt_hash":
        manifest["references"][0]["pixel_sha256"] = "0"*64
    elif change == "version":
        manifest["version"] = True
    elif change == "label":
        manifest["references"][0]["label"] = 1
    elif change == "extra":
        entries["unexpected.png"] = entries[manifest["references"][0]["file"]]
    elif change == "repeat":
        manifest["references"][1] = manifest["references"][0]
    else:
        manifest["references"][0]["file"] = None
    entries["task.json"] = json.dumps(manifest).encode()
    with pytest.raises(ValueError):
        tasks.load_task(archive(entries.items()))


@pytest.mark.parametrize("name", ["../outside.png", "/absolute.png", "C:/file.png", "folder\\file.png"])
def test_batch_zip_paths_are_checked_without_extracting_files(name):
    # ZipInfo normalizes Windows separators when constructing an archive, so
    # preserve the deliberately invalid stored path in both ZIP headers.
    raw = archive([(name.replace("\\", "/"), b"pixels")])
    if "\\" in name:
        raw = raw.replace(name.replace("\\", "/").encode(), name.encode())
    with pytest.raises(ValueError, match="unsafe"):
        tasks.read_images([("batch.zip", raw)])


def test_archive_limits_links_and_duplicate_paths_are_rejected(monkeypatch):
    link = ZipInfo("link.png")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="symbolic"):
        tasks.read_images([("links.zip", archive([(link, b"target")]))])
    with pytest.warns(UserWarning):
        repeated = archive([("same.png", b"one"), ("same.png", b"two")])
    with pytest.raises(ValueError, match="duplicate"):
        tasks.read_images([("repeat.zip", repeated)])
    monkeypatch.setattr(tasks, "MAX_BYTES", 300)
    with pytest.raises(ValueError, match="100 MB"):
        tasks.read_images([("too-big.png", b"x"*301)])


def test_invalid_files_do_not_discard_valid_items_and_reference_errors_are_explicit():
    pixels = np.zeros((12, 24, 4), np.uint8)
    pixels[:, 12:, 3] = 255
    entries = [("good.png", encoded(pixels)), ("broken.png", b"not an image"), ("notes.txt", b"ignored")]
    items, skipped = tasks.read_images([("batch.zip", archive(entries))])
    assert skipped == ["notes.txt"] and len(items) == 2
    assert items[0].error is None and items[1].image is None
    assert np.all(items[0].image[:, :12] == 255)  # Transparent pixels use visible white.
    task = drawing_task()
    with pytest.raises(ValueError, match="repeated"):
        tasks.ImageTask("Repeat", task.groups, "drawings", (task.positive[0],)*2, task.negative)
    with pytest.raises(ValueError, match="both groups"):
        tasks.ImageTask("Conflict", task.groups, "drawings", task.positive, task.positive)


def test_extremely_narrow_image_is_a_visible_item_error_instead_of_breaking_a_batch():
    items, _ = tasks.read_images([("panorama.png", encoded(np.zeros((4, 4000, 3), np.uint8)))])
    assert items[0].image is None and "too narrow" in items[0].error


def test_saved_task_reference_budget_is_bounded_before_serialization(monkeypatch):
    monkeypatch.setattr(tasks, "MAX_REFERENCE_PIXELS", 1)
    with pytest.raises(ValueError, match="30 million"):
        drawing_task()


def test_real_drawing_sort_review_and_export_preserve_originals_and_automatic_evidence():
    task = drawing_task()
    fitted = tasks.fit_task(task)
    assert fitted.diagnostics["selected_pipeline"] == "executable_structural_rules"
    a, b, duplicate = [(FIXTURES/name).read_bytes() for name in ("A05.png", "B05.png", "A01.png")]
    files = [("folder/photo.png", a), ("other/photo.png", b), ("reference.png", duplicate), ("=unsafe.png", b"bad pixels")]
    items, _ = tasks.read_images(files)
    rows = tasks.sort_batch(fitted, items)
    assert [row["automatic_label"] for row in rows] == [True, False, None, None]
    assert "reference image" in rows[2]["reason"] and rows[3]["error"]
    reviews = {"Q0001": False, "Q0003": True, "Q0004": True}
    data = tasks.export_results(task, fitted, items, rows, reviews)
    assert rows[0]["automatic_label"] is True  # Reviews never rewrite automatic evidence.
    with ZipFile(BytesIO(data)) as result:
        report = json.loads(result.read("results.json"))
        assert [r["final_label"] for r in report["results"]] == [False, False, True, None]
        assert [r["decision_source"] for r in report["results"]] == ["human", "automatic", "human", "error"]
        for record, item in zip(report["results"], items):
            assert result.read(record["export_path"]) == item.data
            assert sha256(result.read(record["export_path"])).hexdigest() == record["original_sha256"]
        table = list(csv.DictReader(result.read("results.csv").decode("utf-8-sig").splitlines()))
        assert table[-1]["filename"] == "'=unsafe.png"
        assert report["results"][-1]["filename"] == "=unsafe.png"
        assert tasks.load_task(result.read("task.zip")).fingerprint == task.fingerprint
        assert report["fit"]["query_labels_used"] is False
    items[0].data = b"changed"
    with pytest.raises(ValueError, match="changed"):
        tasks.export_results(task, fitted, items, rows)


def test_imported_task_refits_to_same_decisions_and_invalid_reviews_are_rejected():
    task = drawing_task()
    reopened = tasks.load_task(task.export())
    fitted = tasks.fit_task(reopened)
    items, _ = tasks.read_images([("unseen.png", (FIXTURES/"A06.png").read_bytes())])
    rows = tasks.sort_batch(fitted, items)
    assert rows[0]["automatic_label"] is True
    with pytest.raises(ValueError, match="Reviews"):
        tasks.export_results(task, fitted, items, rows, {"not-in-this-batch": False})
    with pytest.raises(ValueError, match="Reviews"):
        tasks.export_results(task, fitted, items, rows, {rows[0]["id"]: 1})
