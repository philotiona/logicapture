"""Whole-index coverage audit; never equate a support fit with intended-rule recovery."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from src.vision.pretrained import rgb_image
from src.vision.panel_detection import split_problem_page
from src.reasoning.drawing import fit_drawing_rules


def _audit_image(job):
    root,record,media = job
    cv2.setNumThreads(1)
    result = {"id":record["id"],"number":record["number"],"image_sha256":media["sha256"],
              "source_url":media["source_url"],"status":"unresolved"}
    start = time.perf_counter()
    try:
        if media["frames"] != 1:
            return {**result,"reason":"animation_requires_temporal_reasoning"}
        path = Path(root)/media["file"]
        if sha256(path.read_bytes()).hexdigest() != media["sha256"]:
            raise ValueError("Source hash changed")
        with Image.open(path) as image:
            pixels = np.array(rgb_image(image))
        _,panels,layout = split_problem_page(pixels)
        positive = [p.image for p in panels if p.label]
        negative = [p.image for p in panels if not p.label]
        result["layout"] = layout
        if min(len(positive),len(negative)) < 4:
            return {**result,"reason":"not_enough_supports_for_withheld_rule_check"}
        # Freeze these query cells BEFORE fitting; group names and source IDs
        # never cross the inference boundary. Query answers are applied below.
        queries,truth = [positive[-1],negative[-1]],[True,False]
        if any(np.array_equal(q,s) for q in queries for s in positive[:-1]+negative[:-1]):
            return {**result,"reason":"query_duplicates_support_pixels"}
        fit,attempt,pipeline = fit_drawing_rules(positive[:-1],negative[:-1])
        result.update(pipeline=pipeline,rule_check={k:v for k,v in attempt.items() if k!="support_evidence"})
        if fit is None:
            result["reason"] = attempt["reason"]
        else:
            predictions = fit.predict(queries,("Q01","Q02"))
            result.update(status="executable_candidate",rule=attempt["candidates"][0]["text"],
                queries=[{"label":p.label,"truth":truth[i],"evidence":p.diagnostics} for i,p in enumerate(predictions)],
                correct=sum(p.label is label for p,label in zip(predictions,truth)),
                answered=sum(p.label is not None for p in predictions))
    except Exception as error:
        result["reason"] = "layout_or_inference_error"
        result["error"] = str(error)
    result["seconds"] = time.perf_counter()-start
    return result


def audit_foundalis(root, output, *, workers=4):
    root,output = Path(root).resolve(),Path(output).resolve()
    manifest_bytes = (root/"manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    source_files = sorted(Path("src").rglob("*.py"))
    source_hashes = {str(path).replace("\\","/"):sha256(path.read_bytes()).hexdigest() for path in source_files}
    jobs = [(str(root),record,media) for record in manifest["items"] for media in record["images"]]
    results = []
    failures = [{"id":r["id"],"status":"source_unavailable","reason":r.get("error")} for r in manifest["items"] if not r["images"]]
    report = {"index_url":manifest["source_url"],"index_sha256":manifest["index_sha256"],
        "manifest_sha256":sha256(manifest_bytes).hexdigest(),"source_hashes":source_hashes,
        "numbered_entries":len(manifest["items"]),"available_entries":manifest["available"],
        "image_count":len(jobs),"policy":"Final cell per group withheld before fitting. No solutions, captions, concept names or filenames enter inference.",
        "intended_rule_accuracy":None,"limitation":"Executable candidates and query accuracy do not establish recovery of every intended rule.",
        "results":failures}
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        report["results"] = sorted(results+failures,key=lambda r:(r["id"],r.get("image_sha256","")))
        report["complete"] = len(results) == len(jobs)
        report["summary"] = {"processed_images":len(results),"split_images":sum("layout" in r for r in results),
            "executable_candidates":sum(r["status"]=="executable_candidate" for r in results),
            "answered_queries":sum(r.get("answered",0) for r in results),
            "correct_queries":sum(r.get("correct",0) for r in results),
            "fully_correct_query_pairs":sum(r.get("correct")==2 for r in results)}
        # Single writer, fresh file; reports can be resumed at the source cache.
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(report,indent=2),encoding="utf-8")
        for attempt in range(8):
            try:
                temp.replace(output)
                break
            except PermissionError:
                if attempt==7:
                    raise
                time.sleep(.1*(attempt+1))
    save()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_audit_image,job) for job in jobs]
        for task in as_completed(futures):
            results.append(task.result())
            if len(results)%10==0 or len(results)==len(jobs):
                save()
                print(f"{len(results)}/{len(jobs)} images audited; {report['summary']}",flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",default="data/external/foundalis")
    parser.add_argument("--output",default="outputs/foundalis_audit.json")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    audit_foundalis(args.root,args.output,workers=args.workers)
