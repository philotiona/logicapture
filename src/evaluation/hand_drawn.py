"""Evaluate actual cropped drawings without inventing human-data results."""
import argparse
import csv
import json
from pathlib import Path
from src.dsl.nodes import Rule
from src.dsl.verbalizer import verbalize
from src.scene import Scene
from src.synthesis import search
from src.vision import load_image, extract_scene
from .experiments import _perception_metrics


def evaluate_manifest(manifest_path, output="outputs/hand_drawn", adaptive=False, color=False,
                      attributes_model=None, representation="adaptive", remove_panel_ruling=True):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records, errors = [], []
    model = None
    if attributes_model:
        from src.vision.attribute_cnn import load_attribute_model
        model = load_attribute_model(attributes_model)
    if not manifest.get("problems"): raise ValueError("Manifest contains no problems")
    for problem in manifest["problems"]:
        problem_id = problem["id"]
        try:
            examples, names = [], []
            target = Rule.from_dict(problem["target"]) if problem.get("target") else None
            for key, label in (("group_a", True), ("group_b", False)):
                paths = problem[key]
                if len(paths) < 2: raise ValueError("At least two real panels per group are required for holdout")
                for path in paths:
                    full_path = manifest_path.parent / path
                    if not full_path.is_file(): raise FileNotFoundError(f"Drawing is missing: {full_path}")
                    examples.append((extract_scene(load_image(full_path), adaptive=adaptive,
                                                   attribute_model=model, uncertain_only=False,
                                                   representation=representation, remove_panel_ruling=remove_panel_ruling), label))
                    names.append(path)
            heldout = {len(problem["group_a"])-1, len(examples)-1}
            result = search([item for i, item in enumerate(examples) if i not in heldout], color=color,
                            paths=representation == "adaptive")
            best = result.ranked[0].rule
            accepted = result.ranked[0].correct == result.ranked[0].total and result.ranked[0].unknown == 0
            panels = []
            for i, (scene, label) in enumerate(examples):
                annotation = problem.get("annotations", {}).get(names[i])
                metric = _perception_metrics(Scene.from_dict(annotation), scene) if annotation else None
                panels.append({"path": names[i], "label": label, "held_out": i in heldout,
                               "prediction": best.assess(scene) if accepted else None, "scene": scene.to_dict(), "perception": metric})
            predictions = {i: best.assess(examples[i][0]) if accepted else None for i in heldout}
            records.append({"id": problem_id, "artist": problem.get("artist"), "lighting": problem.get("lighting"),
                            "accuracy": sum(predictions[i] is examples[i][1] for i in heldout)/len(heldout),
                            "coverage": sum(predictions[i] is not None for i in heldout)/len(heldout),
                            "accepted_rule": accepted, "unresolved_support": result.ranked[0].unknown,
                            "support_accuracy": result.ranked[0].accuracy,
                            "exact_program_recovery": best == target if target else None,
                            "explanation": verbalize(best), "perfect_hypotheses": result.perfect_count,
                            "search_seconds": result.seconds, "panels": panels})
        except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
            errors.append({"id": problem_id, "error": str(error)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"source_manifest": str(manifest_path), "real_problems_completed": len(records),
              "attributes_model": str(attributes_model) if attributes_model else None,
              "representation": representation,
              "errors": errors, "records": records,
              "annotation_note": "Object and relation metrics require independent annotations; missing annotations remain null."}
    (output/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = ["id", "artist", "lighting", "accuracy", "coverage", "accepted_rule", "support_accuracy", "unresolved_support", "exact_program_recovery", "explanation", "perfect_hypotheses", "search_seconds"]
    with (output/"results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", default="outputs/hand_drawn")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--attributes-model", default=None)
    parser.add_argument("--representation", choices=("adaptive", "geometric"), default="adaptive")
    parser.add_argument("--keep-ruling", action="store_true")
    args = parser.parse_args()
    report = evaluate_manifest(args.manifest, args.output, args.adaptive, args.color, args.attributes_model,
                               args.representation, not args.keep_ruling)
    print(json.dumps({"completed": report["real_problems_completed"], "errors": report["errors"]}, indent=2))
    if report["errors"]: raise SystemExit(1)
