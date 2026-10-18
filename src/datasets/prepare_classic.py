"""Prepare fixed-layout classic research pages; never run external crop code."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from PIL import Image

from .episodes import DatasetError, resolve_image


def prepare_classic_pages(root, *, page_names=None, split="classic-first20"):
    root = Path(root).resolve()
    page_names = list(page_names) if page_names is not None else sorted(p.name for p in root.glob("p????.png"))
    if not page_names:
        raise DatasetError("No original classic Bongard pages were found")
    records = []
    for name in page_names:
        if not re.fullmatch(r"p\d{4}\.png", name):
            raise DatasetError("Unexpected classic page filename")
        source = resolve_image(root, name, require_file=True)
        digest = sha256(source.read_bytes()).hexdigest()
        with Image.open(source) as page:
            if page.size != (2234, 1409):
                raise DatasetError("Classic crop layout requires the original2234×1409 OSF pages")
            page = page.convert("RGB")
            groups = []
            boxes = []
            for label, start_x in (("positive", 9), ("negative", 1312)):
                paths = []
                for row in range(3):
                    for column in range(2):
                        x, y = start_x+column*480, 9+row*480
                        box = (x, y, x+430, y+430)
                        relative = f"panels/{source.stem}/{label}_{row*2+column}.png"
                        target = resolve_image(root, relative)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        page.crop(box).save(target)
                        paths.append(relative)
                        boxes.append({"file": relative, "source_bbox": box})
                groups.append(paths)
        records.append({"id": source.stem, "positive": groups[0], "negative": groups[1],
                        "source_page": name, "source_page_sha256": digest, "crops": boxes})
    manifest = {"format": "classic-bongard-cells-v1", "dataset": "classic-bongard",
                "split": split, "source_url": "https://osf.io/95dks/",
                "selection": "Page names fixed before model evaluation; originals and crops retained locally.",
                "crop_protocol": "430px interiors; fixed OSF layout, row-major; final cell each group held out.",
                "items": records}
    (root/"episodes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--split", default="classic-first20")
    args = parser.parse_args()
    report = prepare_classic_pages(args.root, split=args.split)
    print(f"Prepared {len(report['items'])} problems into {Path(args.root)/'episodes.json'}")
