from dataclasses import replace
from pathlib import Path

from PIL import Image

from src.datasets import DatasetEpisode, ImageRef, Query, SplitProvenance
from src.evaluation.general_benchmark import input_manifest, summarize, refresh_summary, _save
from src.evaluation.parallel_benchmark import fit_without_query_truth
from src.reasoning import FeatureBatch, EpisodeClassifier, EpisodeSupport
import numpy as np


def episode(root):
    refs = []
    for i in range(6):
        name = f"{i}.png"
        Image.new("RGB", (10, 10), (i*30, i*30, i*30)).save(root/name)
        refs.append(ImageRef(root, name))
    return DatasetEpisode("unit", "one", "test", tuple(refs[:2]), tuple(refs[2:4]),
                          (Query(refs[4], True), Query(refs[5], False)),
                          SplitProvenance("https://example.org/data", "digest", "test"))


def test_input_freeze_captures_pixel_crop_and_provenance_mutations(tmp_path):
    item = episode(tmp_path)
    first = input_manifest([item])
    Image.new("RGB", (10, 10), "red").save(tmp_path/"4.png")
    assert input_manifest([item]) != first
    modified = replace(item, support_positive=(replace(item.support_positive[0], crop_bbox=(1, 1, 5, 5)), item.support_positive[1]))
    assert input_manifest([modified]) != input_manifest([item])
    assert input_manifest([replace(item, provenance=replace(item.provenance, annotation_sha256="other"))]) != input_manifest([item])


def test_summary_reports_prototype_and_same_eligible_denominator():
    result = {"metrics": {"total": 2, "answered": 1, "correct": 1, "forced_correct_credit": 1.5},
              "prototype_correct_credit": 1}
    summary = summarize([{"methods": {"hybrid": result}}])["hybrid"]
    assert summary["coverage"] == .5
    assert summary["selective_accuracy"] == 1
    assert summary["forced_accuracy"] == .75
    assert summary["global_prototype_forced_accuracy"] == .5


def test_full_cohort_coverage_keeps_availability_and_strict_exclusions_visible():
    report = {"records": [{"methods": {"hybrid": {"metrics": {"total": 2, "answered": 1, "correct": 1}}}}],
              "excluded": [{"episode_id": "blank"}], "cohort": {"planned_problems": 4, "available_problems": 3}}
    refresh_summary(report)
    assert report["cohort"]["evaluated_problems"] == 1
    assert report["cohort"]["strict_preparation_excluded"] == 1
    assert report["cohort"]["pending_available_problems"] == 1
    assert report["summary"]["hybrid"]["full_planned_cohort_answer_coverage"] == .125


def test_cpu_worker_matches_direct_fit_and_accepts_no_query_truth():
    batch = FeatureBatch(tuple(f"row{i}" for i in range(8)), np.array([[i, 1] for i in (0, 1, 2, 8, 9, 10, 1.5, 8.5)]), encoder_id="unit")
    labels = (True, True, True, False, False, False)
    result = fit_without_query_truth({"structural": batch}, ["structural"], 6, labels)["structural"]
    fitted = EpisodeClassifier().fit(EpisodeSupport(batch.subset(range(6)), labels))
    direct = fitted.predict(batch.subset([6, 7]))
    assert [p.to_dict() for p in result["predictions"]] == [p.to_dict() for p in direct]
    assert "query_truth" not in result


def test_checkpoint_retries_transient_windows_reader_lock(monkeypatch, tmp_path):
    import json
    import src.evaluation.general_benchmark as benchmark
    target = tmp_path/"checkpoint.json"
    target.write_text('{"old": true}', encoding="utf-8")
    original, attempts = Path.replace, []
    def intermittent(self, destination):
        attempts.append(True)
        if len(attempts) < 3:
            assert json.loads(target.read_text()) == {"old": True}
            raise PermissionError("Simulated Windows reader lock")
        return original(self, destination)
    monkeypatch.setattr(Path, "replace", intermittent)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)
    _save(target, {"complete": True})
    assert len(attempts) == 3
    assert json.loads(target.read_text()) == {"complete": True}
