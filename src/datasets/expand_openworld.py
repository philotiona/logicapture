"""Predeclare an expanded public-image cohort without modifying an earlier run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from .episodes import DatasetError
from .fetch_openworld import UPSTREAM_COMMIT, fetch_openworld_subset, freeze_openworld_cohort


def expand_openworld(old_root, new_root, *, val_count=20, test_count=50):
    old_root, new_root = Path(old_root).resolve(), Path(new_root).resolve()
    if old_root == new_root or new_root.is_relative_to(old_root):
        raise DatasetError("Use a separate expansion directory to retain the original frozen run")
    if type(val_count) is not int or type(test_count) is not int or not 1 <= val_count <= 50 or not 1 <= test_count <= 50:
        raise DatasetError("Expansion counts must be bounded1–50")
    new_root.mkdir(parents=True, exist_ok=True)
    request_file = new_root/"expansion_request.json"
    if request_file.exists():
        raise DatasetError("Expansion request already exists; preserve this frozen data run")
    request = {"format": "openworld-expansion-request-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_commit": UPSTREAM_COMMIT, "validation_count": val_count, "test_count": test_count,
               "selection": "First requested episodes in official split order; fixed before retrieval or model inference.",
               "eligibility": "Original final query in each group required; first three available original support images per group; exact duplicates rejected.",
               "original_cache": str(old_root), "original_cohorts": {},
               "no_model_performance_used_for_selection": True}
    for name in ("cohort_dev.json", "cohort_test.json"):
        path = old_root/name
        if path.is_file():
            request["original_cohorts"][name] = sha256(path.read_bytes()).hexdigest()
    # Persist the selection before any network call.
    request_file.write_text(json.dumps(request, indent=2), encoding="utf-8")
    for name in ("bongard_ow.json", "bongard_ow_val.json", "bongard_ow_test.json"):
        source = old_root/name
        if source.is_file():
            (new_root/name).write_bytes(source.read_bytes())
    reused = 0
    source_images = old_root/"images_portable"
    target_images = new_root/"images_portable"
    target_images.mkdir(exist_ok=True)
    if source_images.is_dir():
        for source in source_images.glob("*.img"):
            # Cache names are opaque SHA256 keys, never external path strings.
            if len(source.stem) != 64 or any(c not in "0123456789abcdef" for c in source.stem):
                continue
            target = target_images/source.name
            if target.exists():
                continue
            try:
                os.link(source, target)
            except OSError:
                target.write_bytes(source.read_bytes())
            reused += 1
    print(f"Expansion request frozen; reused {reused} cached images. No earlier cohort was modified.", flush=True)
    def retrieve(split, count):
        index = fetch_openworld_subset(new_root, split=split, count=count, workers=5)
        cohort = freeze_openworld_cohort(new_root, split=split, support_per_group=3,
                                        output_name=f"cohort_{'dev' if split=='val' else 'test'}_expanded.json")
        print(f"{split}: {index['summary']}; eligible {len(cohort['episode_ids'])}/{count}", flush=True)
        return {"split": split, "download_summary": index["summary"], "eligible": len(cohort["episode_ids"]),
                "attempted": count, "excluded": cohort["excluded"]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(retrieve, split, count) for split, count in (("val",val_count),("test",test_count))]
        results = [job.result() for job in jobs]
    report = {"request_sha256": sha256(request_file.read_bytes()).hexdigest(), "reused_cached_images": reused,
              "results": results}
    (new_root/"expansion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("old_root")
    parser.add_argument("new_root")
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=50)
    args = parser.parse_args()
    expand_openworld(args.old_root, args.new_root, val_count=args.val_count, test_count=args.test_count)
