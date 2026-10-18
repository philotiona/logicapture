"""Bounded CPU workers; one main-process frozen encoder owns the GPU."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import time

from src.reasoning import EpisodeSupport, EpisodeClassifier, fuse_feature_batches, nearest_prototype
from src.reasoning.evaluation import prediction_metrics


def fit_without_query_truth(batches, methods, count, labels):
    results = {}
    for method in methods:
        batch = fuse_feature_batches(batches) if method == "hybrid" else batches[method]
        support = EpisodeSupport(batch.subset(range(count)), labels)
        query = batch.subset(range(count, len(batch.image_ids)))
        fitted = EpisodeClassifier().fit(support)
        results[method] = {"predictions": fitted.predict(query), "fit": fitted.diagnostics,
                           "baseline": nearest_prototype(support, query).tolist(), "encoder_id": batch.encoder_id}
    return results


def run_parallel(args, episodes, output, report, completed):
    from .general_benchmark import _save, refresh_summary
    from src.datasets import DatasetError
    from src.vision.pretrained import SiglipEncoder
    from src.vision.structural_features import StructuralEncoder
    semantic = SiglipEncoder() if set(args.methods)&{"semantic", "hybrid"} else None
    structural = StructuralEncoder() if set(args.methods)&{"structural", "hybrid"} else None
    pending = deque()
    report["environment"]["cpu_workers"] = args.jobs

    def finish_one():
        future, row, truth, started, index = pending.popleft()
        values = future.result()
        for method, item in values.items():
            predictions = item.pop("predictions")
            baseline = item.pop("baseline")
            # Truth never crossed the worker's fitting/prediction boundary.
            metrics = prediction_metrics(predictions, truth)
            metrics["correct"] = sum(p.label is label for p, label in zip(predictions, truth))
            metrics["forced_correct_credit"] = metrics["forced_accuracy"]*len(predictions)
            item.update(metrics=metrics, predictions=[p.to_dict() for p in predictions],
                        prototype_correct_credit=sum(.5 if abs(score)<1e-10 else float(bool(score>0) is label)
                                                     for score, label in zip(baseline, truth)))
            row["methods"][method] = item
            print(f"{index+1}/{len(episodes)} {row['episode_id']} {method}: {metrics['correct']}/{metrics['total']} answered={metrics['answered']} forced={metrics['forced_accuracy']:.2f}", flush=True)
        row["seconds_including_queue"] = time.perf_counter()-started
        report["records"].append(row)
        refresh_summary(report)
        _save(output, report)

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for index, episode in enumerate(episodes):
            if episode.episode_id in completed: continue
            started = time.perf_counter()
            try: prepared = episode.prepare(seed=args.seed)
            except DatasetError as error:
                report["excluded"].append({"episode_id": episode.episode_id, "reason": str(error)})
                refresh_summary(report); _save(output, report)
                continue
            data = prepared.input
            images = data.positive+data.negative+data.queries
            count = len(data.positive)+len(data.negative)
            ids = tuple(f"A{i+1:02d}" for i in range(len(data.positive)))+tuple(f"B{i+1:02d}" for i in range(len(data.negative)))+tuple(f"Q{i+1:02d}" for i in range(len(data.queries)))
            labels = (True,)*len(data.positive)+(False,)*len(data.negative)
            batches = {}
            if semantic: batches["semantic"] = semantic.encode(images, ids)
            if structural: batches["structural"] = structural.encode(images).as_feature_batch(ids)
            row = {"episode_id": episode.episode_id, "split": episode.split, "provenance": asdict(episode.provenance),
                   "query_order": prepared.query_order, "query_truth": prepared.query_labels, "methods": {}}
            future = pool.submit(fit_without_query_truth, batches, args.methods, count, labels)
            pending.append((future, row, prepared.query_labels, started, index))
            if len(pending) >= args.jobs*2 or pending[0][0].done(): finish_one()
        while pending: finish_one()
    report["complete"] = True
    refresh_summary(report); _save(output, report)
