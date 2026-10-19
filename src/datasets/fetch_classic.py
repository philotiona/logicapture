"""Fetch small public OSF classic pages and freeze development/test manifests."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from .episodes import DatasetError
from .fetch_openworld import _get
from .prepare_classic import prepare_classic_pages

SOURCE = "https://api.osf.io/v2/nodes/95dks/files/osfstorage/65c674103280d80d5da3aa33/?page[size]=100"


def fetch_classic_pages(root, *, count=100, development_count=20):
    if type(count) is not int or not 1 <= count <= 100 or not 1 <= development_count < count:
        raise DatasetError("Choose2–100 pages and a nonempty development/test partition")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(_get(SOURCE, 2_000_000, 30))
    files = sorted(metadata["data"], key=lambda item: item["attributes"]["name"])[:count]
    expected_names = [f"p{i:04d}.png" for i in range(1, count+1)]
    if [item["attributes"]["name"] for item in files] != expected_names:
        raise DatasetError("OSF page names do not match the declared consecutive cohort")
    (root/"osf_files.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    def download(item):
        name = item["attributes"]["name"]
        target = root/name
        payload = target.read_bytes() if target.is_file() else _get(item["links"]["download"], 2_000_000, 30)
        digest = sha256(payload).hexdigest()
        expected = item["attributes"]["extra"]["hashes"].get("sha256")
        if expected and digest != expected:
            raise DatasetError(f"OSF checksum mismatch: {name}")
        with Image.open(BytesIO(payload)) as image:
            if image.size != (2234, 1409):
                raise DatasetError("Unexpected OSF classic page dimensions")
            image.verify()
        target.write_bytes(payload)
        return {"file": name, "sha256": digest, "bytes": len(payload),
                "source_url": item["links"]["download"], "osf_id": item["id"]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(download, files))
    manifest = prepare_classic_pages(root, page_names=expected_names, split=f"classic-all{count}")
    for filename, items, split in (("episodes_dev.json", manifest["items"][:development_count], f"classic-dev-001-{development_count:03d}"),
                                    ("episodes_test.json", manifest["items"][development_count:], f"classic-test-{development_count+1:03d}-{count:03d}")):
        (root/filename).write_text(json.dumps({**manifest, "items": items, "split": split}, indent=2), encoding="utf-8")
    report = {"source": "https://osf.io/95dks/", "files": records,
              "selection": f"First{development_count} pages development; remaining{count-development_count} pages test, fixed before inference.",
              "license": "OSF node_license metadata was null; no redistribution license inferred. Images retained locally only."}
    (root/"download_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--development-count", type=int, default=20)
    args = parser.parse_args()
    report = fetch_classic_pages(args.root, count=args.count, development_count=args.development_count)
    print(f"Prepared{len(report['files'])} pages; evaluation must report duplicate exclusions separately.")
