"""Reproducible, local, pixel-only broad Bongard evaluation.

This measures held-out image classification, not recovery of the intended rule.
No dataset solution text or query truth is sent to the encoders/classifier/VLM.
Records are saved after each problem, with a configuration and input-content
fingerprint for safe resume. Supports are fitted within each problem; there is
no cross-problem fine-tuning and no fitting to query labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import time

import numpy as np

from src.datasets import DatasetError, load_classic, load_hoi, load_logo, load_openworld, load_openworld_cohort
from src.reasoning import EpisodeSupport, EpisodeClassifier, fuse_feature_batches, nearest_prototype
from src.reasoning.evaluation import prediction_metrics
from src.vision.pretrained import SiglipEncoder, QwenLocalBackend, MODELS
from src.vision.structural_features import StructuralEncoder


def _save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows readers/indexers can briefly hold a file without delete
            # sharing. Keep the previous valid checkpoint until replace works.
            if attempt == 7: raise
            time.sleep(min(.05*2**attempt, .5))


def summarize(records):
    names = sorted({name for row in records for name in row.get("methods", {})})
    summary = {}
    for name in names:
        results = [row["methods"][name] for row in records if name in row.get("methods", {})]
        total = sum(item["metrics"]["total"] for item in results)
        answered = sum(item["metrics"]["answered"] for item in results)
        correct = sum(item["metrics"]["correct"] for item in results)
        forced = sum(item["metrics"].get("forced_correct_credit", 0) for item in results)
        prototype = sum(item.get("prototype_correct_credit", 0) for item in results)
        summary[name] = {"problems": len(results), "queries": total, "answered": answered,
                         "correct": correct, "coverage": answered/total if total else 0,
                         "accuracy_including_abstentions": correct/total if total else None,
                         "selective_accuracy": correct/answered if answered else None,
                         "forced_accuracy": forced/total if total and name != "language" else None,
                         "global_prototype_forced_accuracy": prototype/total if total and name != "language" else None,
                         "denominator": "eligible evaluated queries only; see cohort counts/exclusions",
                         "errors": sum("error" in item for item in results)}
    return summary


def input_manifest(episodes):
    files, planned = {}, []
    for episode in episodes:
        images = []
        for ref in episode.images:
            path = ref.path
            if path not in files: files[path] = sha256(path.read_bytes()).hexdigest()
            images.append({"relative_path": ref.relative_path, "sha256": files[path],
                           "crop_bbox": ref.crop_bbox, "coordinate_scale": ref.coordinate_scale})
        planned.append({"dataset": episode.dataset, "episode_id": episode.episode_id, "split": episode.split,
                        "provenance": asdict(episode.provenance), "images": images,
                        "positive_support_count": len(episode.support_positive),
                        "negative_support_count": len(episode.support_negative),
                        "query_truth": [query.label for query in episode.queries]})
    return planned


def configuration(args, manifest):
    root = Path(__file__).resolve().parents[2]
    source_files = ["src/reasoning/episode.py", "src/reasoning/features.py", "src/reasoning/concepts.py",
                    "src/vision/pretrained.py", "src/vision/structural_features.py", "src/datasets/adapters.py",
                    "src/datasets/episodes.py", "src/reasoning/evaluation.py", "src/evaluation/general_benchmark.py",
                    "src/reasoning/grounding.py", "src/evaluation/parallel_benchmark.py"]
    return {"dataset": args.dataset, "partition": args.partition, "limit": args.limit, "seed": args.seed, "jobs": args.jobs,
            "methods": args.methods, "models": {k: asdict(v) for k, v in MODELS.items()},
            "source_sha256": {file: sha256((root/file).read_bytes()).hexdigest() for file in source_files},
            "support_only_nested_selection": True, "query_labels_used_for_fitting": False,
            "pretrained_model_contamination": "Unknown; external pretraining may include classic concepts/images.",
            "measurement": "within-problem heldout image classification, not logical rule recovery",
            "language_max_candidates": args.language_candidates, "language_model": args.language_model,
            "language_max_rounds": args.language_rounds,
            "language_measured_grounding": True,
            "cohort_sha256": sha256(Path(args.cohort).read_bytes()).hexdigest() if args.cohort else None,
            "dataset_root_sha256": sha256(str(Path(args.root).resolve()).encode()).hexdigest(),
            "input_manifest_sha256": sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "local_only": True}


def refresh_summary(report):
    report["summary"] = summarize(report["records"])
    cohort = report["cohort"]
    cohort["evaluated_problems"] = len(report["records"])
    cohort["strict_preparation_excluded"] = len(report["excluded"])
    cohort["pending_available_problems"] = cohort["available_problems"]-len(report["records"])-len(report["excluded"])
    for item in report["summary"].values():
        item["full_planned_cohort_answer_coverage"] = item["answered"]/(2*cohort["planned_problems"]) if cohort["planned_problems"] else 0


def _episodes(args):
    root = Path(args.root)
    if args.dataset == "classic":
        return load_classic(root, manifest_file=root/f"episodes_{args.partition}.json", limit=args.limit)
    if args.dataset == "hoi":
        return load_hoi(args.annotations, root, split=args.partition, limit=args.limit)
    if args.dataset == "logo":
        return load_logo(root, split=args.partition, split_file=args.annotations, limit=args.limit)
    if args.cohort:
        return load_openworld_cohort(root, cohort_file=args.cohort)
    return load_openworld(args.annotations, root, split=args.partition, image_index=args.image_index, limit=args.limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("classic", "hoi", "logo", "openworld"), default="classic")
    parser.add_argument("--root", default="data/external/classic_bongard")
    parser.add_argument("--partition", default="dev")
    parser.add_argument("--annotations")
    parser.add_argument("--image-index")
    parser.add_argument("--cohort")
    parser.add_argument("--methods", nargs="+", choices=("structural", "semantic", "hybrid", "language"), default=["structural", "semantic", "hybrid"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--language-candidates", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--language-model", choices=("qwen", "qwen8b"), default="qwen")
    parser.add_argument("--language-rounds", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--output", default="data/benchmarks/classic_dev.json")
    args = parser.parse_args()
    if args.jobs > 1 and "language" in args.methods:
        parser.error("Language inference is sequential; use --jobs 1.")
    episodes, output = _episodes(args), Path(args.output)
    manifest = input_manifest(episodes)
    config = configuration(args, manifest)
    fingerprint = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("configuration_sha256") != fingerprint:
            raise SystemExit("Existing report uses another configuration. Select a new output path.")
    else:
        cohort = json.loads(Path(args.cohort).read_text(encoding="utf-8")) if args.cohort else {}
        planned_ids = cohort.get("attempted_episode_ids", [e.episode_id for e in episodes])
        report = {"configuration": config, "configuration_sha256": fingerprint, "input_manifest": manifest,
                  "cohort": {"planned_problems": len(planned_ids), "available_problems": len(episodes),
                             "availability_exclusions": cohort.get("excluded", [])},
                  "records": [], "summary": {},
                  "environment": {"python": platform.python_version(), "platform": platform.platform()},
                  "excluded": [], "complete": False}
        _save(output, report)  # Freeze configuration BEFORE the first prediction.
    completed = {row["episode_id"] for row in report["records"]} | {row["episode_id"] for row in report["excluded"]}
    expected = {item["episode_id"]: item for item in manifest}
    for row in report["records"]:
        if row["episode_id"] not in expected or row["provenance"] != expected[row["episode_id"]]["provenance"]:
            raise SystemExit("Existing report episode provenance differs from the frozen input plan.")
    if args.jobs > 1:
        from .parallel_benchmark import run_parallel
        run_parallel(args, episodes, output, report, completed)
        print(json.dumps(report["summary"], indent=2), flush=True)
        return
    semantic = SiglipEncoder() if set(args.methods)&{"semantic", "hybrid"} else None
    structural = StructuralEncoder() if set(args.methods)&{"structural", "hybrid"} else None
    backend = QwenLocalBackend(model=args.language_model) if "language" in args.methods else None
    for index, episode in enumerate(episodes):
        if episode.episode_id in completed: continue
        started = time.perf_counter()
        try:
            prepared = episode.prepare(seed=args.seed)
        except DatasetError as error:
            report["excluded"].append({"episode_id": episode.episode_id, "reason": str(error)})
            _save(output, report)
            print(f"{index+1}/{len(episodes)} {episode.episode_id}: excluded: {error}", flush=True)
            continue
        inputs = prepared.input
        images = inputs.positive+inputs.negative+inputs.queries
        count = len(inputs.positive)+len(inputs.negative)
        ids = tuple(f"A{i+1:02d}" for i in range(len(inputs.positive))) + tuple(f"B{i+1:02d}" for i in range(len(inputs.negative))) + tuple(f"Q{i+1:02d}" for i in range(len(inputs.queries)))
        labels = (True,)*len(inputs.positive)+(False,)*len(inputs.negative)
        row = {"episode_id": episode.episode_id, "split": episode.split, "provenance": asdict(episode.provenance),
               "query_order": prepared.query_order, "query_truth": prepared.query_labels, "methods": {}}
        batches = {}
        if semantic: batches["semantic"] = semantic.encode(images, ids)
        if structural: batches["structural"] = structural.encode(images).as_feature_batch(ids)
        for method in args.methods:
            if method == "language": continue
            batch = fuse_feature_batches(batches) if method == "hybrid" else batches[method]
            support = EpisodeSupport(batch.subset(range(count)), labels)
            fitted = EpisodeClassifier().fit(support)
            query = batch.subset(range(count, len(images)))
            predictions = fitted.predict(query)
            # Truth first enters here, after ALL predictions and model selection.
            metrics = prediction_metrics(predictions, prepared.query_labels)
            metrics["correct"] = sum(item.label is label for item, label in zip(predictions, prepared.query_labels))
            metrics["forced_correct_credit"] = metrics["forced_accuracy"]*len(predictions)
            baseline = nearest_prototype(support, query)
            baseline_correct = sum(.5 if abs(score) < 1e-10 else float(bool(score > 0) is label)
                                   for score, label in zip(baseline, prepared.query_labels))
            row["methods"][method] = {"metrics": metrics, "prototype_correct_credit": baseline_correct,
                                      "predictions": [item.to_dict() for item in predictions], "fit": fitted.diagnostics,
                                      "encoder_id": batch.encoder_id}
            print(f"{index+1}/{len(episodes)} {episode.episode_id} {method}: {metrics['correct']}/{metrics['total']} answered={metrics['answered']} forced={metrics['forced_accuracy']:.2f}", flush=True)
        if backend:
            if semantic: semantic.unload()
            from src.reasoning.concepts import ConceptInducer
            try:
                analysis = ConceptInducer(backend, max_candidates=args.language_candidates, max_rounds=args.language_rounds, measured_grounding=True).solve(
                    inputs.positive, inputs.negative, inputs.queries,
                    progress=lambda text: print(f"{episode.episode_id}: {text}", flush=True))
                answered = sum(item.label is not None for item in analysis.predictions)
                correct = sum(item.label is label for item, label in zip(analysis.predictions, prepared.query_labels))
                row["methods"]["language"] = {"analysis": analysis.to_dict(),
                    "metrics": {"total": len(inputs.queries), "answered": answered, "correct": correct}}
            except (ValueError, RuntimeError) as error:
                row["methods"]["language"] = {"error": str(error), "metrics": {"total": len(inputs.queries), "answered": 0, "correct": 0}}
            if semantic: backend.unload()
            print(f"{index+1}/{len(episodes)} {episode.episode_id} language: {row['methods']['language']['metrics']}", flush=True)
        row["seconds"] = time.perf_counter()-started
        report["records"].append(row)
        refresh_summary(report)
        _save(output, report)
    report["complete"] = True
    refresh_summary(report)
    _save(output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__": main()
