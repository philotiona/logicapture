"""Exercise saved photo tasks and lossless batch export on a fixed local cohort."""
from hashlib import sha256
from io import BytesIO
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
from PIL import Image

from src.datasets import load_openworld_cohort
from src.image_tasks import ImageTask, fit_task, load_task, read_images, sort_batch, export_results
from src.vision.pretrained import SiglipEncoder


def _entry(ref):
    # The dataset cache deliberately uses .img names; give uploads their real
    # media extension without changing the original encoded file bytes.
    with Image.open(ref.path) as image:
        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tiff"}[image.format]
    return ref.path.stem+suffix, ref.path.read_bytes()


def category_check(root, encoder):
    """A declared two-category task, separate from Bongard abstraction tests."""
    records = json.loads((root/"bongard_ow_val.json").read_text(encoding="utf-8"))
    paths = json.loads((root/"image_index_val.json").read_text(encoding="utf-8"))["paths"]
    groups = []
    for uid in ("0371", "0179"):
        record = next(record for record in records if record["uid"] == uid)
        entries = [_entry(SimpleNamespace(path=root/paths[name])) for name in record["imageFiles"][:7] if name in paths]
        items, skipped = read_images(entries)
        if skipped or len(items) < 5 or any(item.error for item in items):
            raise ValueError("The category check needs four references and independent queries per class.")
        groups.append(items)
    task = ImageTask("Penguins or tortoises", ("Penguins", "Tortoises"), "photos",
                     tuple(item.image for item in groups[0][:4]), tuple(item.image for item in groups[1][:4]))
    fitted = fit_task(load_task(task.export()), semantic_encoder=encoder)
    # Interleave the remaining independent sources; ordering is not a label cue.
    labeled = [(item, label) for label, group in ((True, groups[0]), (False, groups[1])) for item in group[4:]]
    labeled.sort(key=lambda pair: pair[0].digest)
    items, truth = [pair[0] for pair in labeled], [pair[1] for pair in labeled]
    rows = sort_batch(fitted, items)
    with ZipFile(BytesIO(export_results(task, fitted, items, rows))) as archive:
        saved = json.loads(archive.read("results.json"))["results"]
        intact = all(archive.read(row["export_path"]) == item.data for row, item in zip(saved, items))
    result = {"scope": "Small practical category-sorting development check; not an independent accuracy benchmark.",
              "selection": "Penguins and tortoises selected by category names before category inference. First four available annotated positive images per category are references; remaining sources are withheld.",
              "source_episode_ids": ["0371", "0179"], "task": task.manifest(), "encoder_id": fitted.encoder_id,
              "queries": [{**row, "truth": label} for row, label in zip(rows, truth)],
              "summary": {"queries": len(rows), "answered": sum(row["automatic_label"] is not None for row in rows),
                          "correct": sum(row["automatic_label"] is label for row, label in zip(rows, truth)),
                          "wrong": sum(row["automatic_label"] is not None and row["automatic_label"] is not label for row, label in zip(rows, truth)),
                          "needs_review": sum(row["automatic_label"] is None for row in rows), "originals_preserved": intact},
              "limitations": "Only three withheld images in two existing development categories. Contexts and pretrained-data overlap may help; this does not establish performance on other categories or personal photos."}
    print("Category check:", result["summary"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/external/bongard_openworld_expanded"))
    parser.add_argument("--output", type=Path, default=Path("data/regressions/image_tasks_photo_pilot.json"))
    args = parser.parse_args()
    manifest = args.root/"cohort_dev_expanded.json"
    episodes = load_openworld_cohort(args.root, cohort_file=manifest)[:5]
    if len(episodes) != 5:
        raise ValueError("The fixed pilot requires the first five available development episodes.")
    encoder = SiglipEncoder()
    report = {"scope": "Photo-task workflow development smoke test, not a new generalization benchmark.",
              "selection": "First five available episodes in the existing frozen development cohort; selected before this run.",
              "planned_ids": [episode.episode_id for episode in episodes],
              "cohort_sha256": sha256(manifest.read_bytes()).hexdigest(),
              "source_hashes": {str(path).replace(chr(92), '/'): sha256(path.read_bytes()).hexdigest()
                                for path in (Path("src/image_tasks.py"), Path("src/reasoning/general.py"),
                                             Path("src/reasoning/episode.py"), Path("src/vision/pretrained.py"))},
              "query_labels_used_for_fit": False, "results": []}
    for episode in episodes:
        episode.validate_files(check_content=True)
        groups = [tuple(item.image for item in read_images([_entry(ref) for ref in refs])[0])
                  for refs in (episode.support_positive, episode.support_negative)]
        task = ImageTask("Photo inspection", ("Matches", "Other"), "photos", *groups)
        restored = load_task(task.export())
        assert restored.fingerprint == task.fingerprint
        fitted = fit_task(restored, semantic_encoder=encoder)
        prepared = episode.prepare()
        queries = [episode.queries[index] for index in prepared.query_order]
        items, _ = read_images([_entry(query.image) for query in queries])
        rows = sort_batch(fitted, items)
        with ZipFile(BytesIO(export_results(task, fitted, items, rows))) as exported:
            records = json.loads(exported.read("results.json"))["results"]
            intact = all(exported.read(row["export_path"]) == item.data for row, item in zip(records, items))
        result = {"id": episode.episode_id, "task_fingerprint": task.fingerprint,
                  "encoder_id": fitted.encoder_id, "originals_preserved": intact,
                  "queries": [{**row, "truth": query.label} for row, query in zip(rows, queries)]}
        report["results"].append(result)
        print(episode.episode_id, "decisions:", [row["automatic_label"] for row in rows], "truth:", [q.label for q in queries], flush=True)
    rows = [row for result in report["results"] for row in result["queries"]]
    report["summary"] = {"tasks": len(episodes), "queries": len(rows),
        "answered": sum(row["automatic_label"] is not None for row in rows),
        "correct": sum(row["automatic_label"] is row["truth"] for row in rows),
        "wrong": sum(row["automatic_label"] is not None and row["automatic_label"] is not row["truth"] for row in rows),
        "needs_review": sum(row["automatic_label"] is None for row in rows),
        "all_originals_preserved": all(result["originals_preserved"] for result in report["results"])}
    report["limitations"] = ["These are existing development images, not the user's personal collection.",
        "Three references per group; query labels are used only after decisions for reporting.",
        "No accuracy gain is claimed from this workflow change. Wrong automatic answers and abstentions are retained."]
    report["category_sorting"] = category_check(args.root, encoder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
