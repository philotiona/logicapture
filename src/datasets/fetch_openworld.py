"""Bounded public-image retrieval for a predeclared OpenWorld subset.

Only data is downloaded. No authentication, API inference, downloaded code,
archive extraction or image-URL instructions are executed. Failures stay in the
manifest; successful episodes are never chosen using model performance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import ipaddress
import json
from pathlib import Path
import socket
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

from .episodes import DatasetError

UPSTREAM_COMMIT = "0462ba15f3be2f7f9a7e6cdd0d24314a1fabb5fe"
UPSTREAM = f"https://raw.githubusercontent.com/rujiewu/Bongard-OpenWorld/{UPSTREAM_COMMIT}/assets/data/bongard-ow/"


def _public_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise DatasetError("Only unauthenticated public HTTP(S) image URLs are allowed")
    if parsed.port not in (None, 80, 443):
        raise DatasetError("Nonstandard image URL port")
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)):
        if not ipaddress.ip_address(result[4][0]).is_global:
            raise DatasetError("Image URL resolves to a non-public address")
    return url


class PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        _public_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _get(url, max_bytes, timeout):
    _public_url(url)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; BongardResearchEvaluation/1.0)"})
    with build_opener(PublicRedirects()).open(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise DatasetError("Image exceeds the configured byte limit")
        payload = response.read(max_bytes+1)
    if len(payload) > max_bytes:
        raise DatasetError("Image exceeds the configured byte limit")
    return payload


def fetch_openworld_subset(root, *, split, count, workers=8, timeout=12, max_image_bytes=3_000_000):
    if split not in ("train", "val", "test") or not 1 <= count <= 50:
        raise DatasetError("Use train/val/test and a bounded1–50 episode count")
    if not 1 <= workers <= 12 or not 1 <= timeout <= 60:
        raise DatasetError("Invalid bounded retrieval configuration")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    role = "dev" if split == "val" else split
    if any(root.glob(f"cohort_{role}*.json")):
        raise DatasetError("A frozen cohort exists; use a new data directory for a new retrieval run")
    all_file, split_file = root/"bongard_ow.json", root/f"bongard_ow_{split}.json"
    for target, name, bound in ((all_file, "bongard_ow.json", 5_000_000),
                                (split_file, f"bongard_ow_{split}.json", 2_000_000)):
        if not target.is_file():
            target.write_bytes(_get(UPSTREAM+name, bound, timeout))
    full = json.loads(all_file.read_bytes())
    selected = json.loads(split_file.read_bytes())[:count]
    source_by_id = {record["uid"]: record for record in full}
    jobs = []
    for record in selected:
        source = source_by_id[record["uid"]]
        if source["imageFiles"] != record["imageFiles"] or len(source["urls"]) != 14:
            raise DatasetError("OpenWorld full/split annotation references disagree")
        jobs.extend((record["uid"], i, name, source["urls"][i]) for i, name in enumerate(source["imageFiles"]))
    image_dir = root/"images_portable"
    image_dir.mkdir(exist_ok=True)
    def download(job):
        uid, index, original, url = job
        relative = f"images_portable/{sha256(original.encode()).hexdigest()}.img"
        target = root/relative
        try:
            cached = target.is_file()
            payload = target.read_bytes() if cached else _get(url, max_image_bytes, timeout)
            if len(payload) > max_image_bytes:
                raise DatasetError("Cached image exceeds the configured byte limit")
            with Image.open(BytesIO(payload)) as image:
                if image.width*image.height > 80_000_000:
                    raise DatasetError("Oversized decoded image")
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                image.convert("RGB").load()
            if not cached:
                target.write_bytes(payload)
            return {"uid": uid, "image_index": index, "original": original, "local": relative,
                    "url": url, "bytes": len(payload), "sha256": sha256(payload).hexdigest()}
        except (OSError, ValueError, DatasetError, Image.DecompressionBombError) as error:
            return {"uid": uid, "image_index": index, "original": original, "url": url,
                    "error": f"{type(error).__name__}: {error}"}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(download, jobs))
    successes = [record for record in records if "error" not in record]
    manifest = {"format": "openworld-portable-index-v1", "split": split,
                "source_url": UPSTREAM+split_file.name, "source_commit": UPSTREAM_COMMIT,
                "annotation_sha256": sha256(split_file.read_bytes()).hexdigest(),
                "selection": f"First{count} episodes in the official{split} annotation order, before model evaluation.",
                "attempted_episode_ids": [item["uid"] for item in selected],
                "paths": {record["original"]: record["local"] for record in successes},
                "records": records,
                "summary": {"attempted_images": len(jobs), "downloaded_images": len(successes),
                            "failed_images": len(records)-len(successes),
                            "downloaded_bytes": sum(record["bytes"] for record in successes)}}
    (root/f"image_index_{split}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def freeze_openworld_cohort(root, *, split, support_per_group=3, output_name=None):
    """Freeze availability before inference; no performance-based selection."""
    from .adapters import load_openworld
    if split not in ("val", "test"):
        raise DatasetError("Freeze a validation or test cohort")
    root = Path(root).resolve()
    index_file = root/f"image_index_{split}.json"
    index = json.loads(index_file.read_bytes())
    available = {}
    for record in index["records"]:
        if "error" not in record:
            available.setdefault(record["uid"], set()).add(record["image_index"])
    eligible, excluded, strict = [], [], []
    for uid in index["attempted_episode_ids"]:
        good, reasons = available.get(uid, set()), []
        if 6 not in good or 13 not in good:
            reasons.append("official query image unavailable")
        if sum(i in good for i in range(6)) < support_per_group or sum(i in good for i in range(7,13)) < support_per_group:
            reasons.append(f"fewer than {support_per_group} available support images in a group")
        if not reasons:
            episode, = load_openworld(root/f"bongard_ow_{split}.json", root, split=split, image_index=index_file,
                episode_ids=[uid], support_per_group=support_per_group, allow_missing_support=True)
            try:
                episode.validate_files(check_content=True)
                eligible.append(uid)
            except DatasetError as error:
                reasons.append(str(error))
        if reasons:
            excluded.append({"uid": uid, "reason": reasons})
        if len(good) == 14:
            strict.append(uid)
    cohort = {"format": "openworld-cohort-v1", "split": split,
              "annotation_file": f"bongard_ow_{split}.json", "image_index": index_file.name,
              "image_index_sha256": sha256(index_file.read_bytes()).hexdigest(),
              "source_annotation_sha256": index["annotation_sha256"], "selection": index["selection"],
              "support_per_group": support_per_group, "allow_missing_support": True,
              "protocol": f"First {support_per_group} available original support images per group; original index6 query remains held out. Availability-filtered condition.",
              "attempted_episode_ids": index["attempted_episode_ids"], "episode_ids": eligible,
              "excluded": excluded, "strict_six_shot_complete_ids": strict,
              "download_summary": index["summary"], "configuration_frozen_before_inference": True}
    from .episodes import resolve_image
    role = "dev" if split == "val" else "test"
    target = resolve_image(root, output_name or f"cohort_{role}.json")
    if target.exists():
        raise DatasetError("A frozen cohort already exists; preserve it for the current run")
    target.write_text(json.dumps(cohort, indent=2), encoding="utf-8")
    return cohort


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--freeze-three-shot-cohort", action="store_true")
    args = parser.parse_args()
    result = fetch_openworld_subset(args.root, split=args.split, count=args.count)
    print(json.dumps(result["summary"], indent=2))
    if args.freeze_three_shot_cohort:
        cohort = freeze_openworld_cohort(args.root, split=args.split)
        print(f"Frozen {len(cohort['episode_ids'])} eligible episodes; {len(cohort['excluded'])} exclusions recorded.")
