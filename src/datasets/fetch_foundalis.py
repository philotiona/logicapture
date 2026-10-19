"""Download the numbered Foundalis index as data, never as solver hints.

The frozen cohort includes every numbered link, including retrieval failures.
Only image pixels and left/right group membership reach inference. No solution
page is fetched. Downloaded media stay in the ignored external-data directory.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import argparse
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

from PIL import Image
from .fetch_openworld import _get

INDEX = "https://www.foundalis.com/res/bps/bpidx.htm"


class MediaLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.images = [], []
        self._href, self._text = None, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            self._href, self._text = attrs.get("href"), []
        if tag == "img" and attrs.get("src"):
            self.images.append(attrs["src"])

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, "".join(self._text).strip()))
            self._href = None


def numbered_problems(html, base=INDEX):
    parser = MediaLinks()
    parser.feed(html)
    found = {}
    for href, text in parser.links:
        url = urljoin(base, href)
        if (text.isdecimal() and urlparse(url).hostname == urlparse(base).hostname
                and urlparse(url).path.lower().endswith((".htm", ".html"))):
            found.setdefault(int(text), url)
    return [{"id": f"foundalis-{number:04d}", "number": number, "source_url": url}
            for number, url in sorted(found.items())]


def fetch_foundalis(root, *, workers=4, progress=None):
    if not 1 <= workers <= 6:
        raise ValueError("Use one to six download workers")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    index_file = root/"index.html"
    if not index_file.exists():
        index_file.write_bytes(_get(INDEX, 2_000_000, 30))
    raw = index_file.read_bytes()
    cohort = numbered_problems(raw.decode("utf-8", errors="replace"))
    if not cohort:
        raise ValueError("No numbered problem links were found")
    plan = {"source_url": INDEX, "index_sha256": sha256(raw).hexdigest(), "items": cohort,
            "policy": "Every numbered index entry; no solution text; no performance-based selection."}
    plan_file = root/"cohort.json"
    if plan_file.exists() and json.loads(plan_file.read_text(encoding="utf-8")) != plan:
        raise ValueError("Cohort changed; use a new directory")
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def fetch(item):
        folder = root/item["id"]
        folder.mkdir(exist_ok=True)
        record_file = folder/"record.json"
        if record_file.exists():
            cached = json.loads(record_file.read_text(encoding="utf-8"))
            if cached.get("status") == "available":
                return cached
        record = {**item, "status": "failed", "images": []}
        try:
            page_file = folder/"page.html"
            if not page_file.exists():
                page_file.write_bytes(_get(item["source_url"], 2_000_000, 20))
            html = page_file.read_bytes()
            parser = MediaLinks()
            parser.feed(html.decode("utf-8", errors="replace"))
            record["page_sha256"] = sha256(html).hexdigest()
            for source in dict.fromkeys(parser.images):
                url = urljoin(item["source_url"], source)
                if urlparse(url).hostname != urlparse(INDEX).hostname:
                    continue
                name = sha256(url.encode()).hexdigest()[:24]+".img"
                target = folder/name
                try:
                    data = target.read_bytes() if target.exists() else _get(url, 12_000_000, 20)
                    with Image.open(BytesIO(data)) as image:
                        width, height = image.size
                        if width*height > 20_000_000:
                            raise ValueError("Oversized source image")
                        frames = getattr(image, "n_frames", 1)
                        image.verify()
                    if width < 72 or height < 72:
                        continue  # navigation and other small decorations
                    target.write_bytes(data)
                    record["images"].append({"file": target.relative_to(root).as_posix(),
                        "source_url": url, "sha256": sha256(data).hexdigest(),
                        "width": width, "height": height, "frames": frames})
                except (ValueError, OSError) as error:
                    record.setdefault("image_errors", []).append({"url": url, "error": str(error)})
            if record["images"]:
                record["status"] = "available"
            else:
                record["error"] = "No accessible problem-size image"
        except (ValueError, OSError) as error:
            record["error"] = str(error)
        record_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(fetch, item): item for item in cohort}
        for task in as_completed(tasks):
            records.append(task.result())
            if progress:
                progress(len(records), len(cohort), records[-1])
    records.sort(key=lambda item: item["number"])
    manifest = {**plan, "items": records,
                "available": sum(item["status"] == "available" for item in records),
                "license": "No redistribution license inferred; media retained locally only."}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    def status(done, total, item):
        if done % 25 == 0 or done == total:
            print(f"{done}/{total} entries retrieved", flush=True)
    result = fetch_foundalis(args.root, workers=args.workers, progress=status)
    print(f"{result['available']}/{len(result['items'])} entries have local images")
