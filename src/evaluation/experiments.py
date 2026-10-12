import argparse
from dataclasses import replace
import json
import csv
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image

from src.dsl.nodes import Predicate, Exists, Count, Relation, And
from src.dsl.verbalizer import verbalize
from src.dsl.grammar import enumerate_rules
from src.active import SceneGenerator, AkinatorSession, GenerationError
from src.scene.renderer import draw_scene, draw_page
from src.scene import Scene, derive_relations
from src.vision import extract_scene
from src.synthesis import search
from src.baselines.pixel import prototype_predict


def targets():
    circle, triangle = Predicate("shape", "circle"), Predicate("shape", "triangle")
    return [Exists(circle), Exists(Predicate("filled")), Count(2),
            And(Exists(circle), Count(2)), Relation("left_of", triangle, circle),
            Relation("inside", Predicate("filled"), Predicate("outline"), universal=True)]


def perturb(image, condition, rng):
    if condition == "noise":
        pixels = np.asarray(image, dtype=np.float32)
        return Image.fromarray(np.uint8(np.clip(pixels+rng.normal(0, 18, pixels.shape), 0, 255)))
    return image


def render_condition(scene, condition, rng):
    # Rotating the complete image changes left/above relations and makes the
    # old scene coordinates invalid for perception scoring. Rotate individual
    # objects instead, retain their centers, and recompute ground-truth geometry.
    truth = derive_relations(Scene([replace(o, angle=o.angle+18) for o in scene.objects])) if condition == "rotation" else scene
    return perturb(draw_scene(truth), condition, rng), truth


def _perception_metrics(truth, detected):
    unmatched = list(detected.objects)
    matches, attribute_matches = {}, 0
    for obj in truth.objects:
        if not unmatched: continue
        nearest = min(unmatched, key=lambda other: (other.center_x-obj.center_x)**2+(other.center_y-obj.center_y)**2)
        if (nearest.center_x-obj.center_x)**2+(nearest.center_y-obj.center_y)**2 < .035**2:
            matches[obj.id] = nearest.id
            attribute_matches += obj.shape == nearest.shape and obj.filled == nearest.filled
            unmatched.remove(nearest)
    true_rel = {r for r in truth.relations if r[0] in ("left_of", "above", "inside", "near", "larger_than")}
    mapped = {(r[0], *[matches[i] for i in r[1:]]) for r in true_rel if all(i in matches for i in r[1:])}
    detected_rel = {r for r in detected.relations if r[0] in ("left_of", "above", "inside", "near", "larger_than")}
    return {"object_count_correct": len(truth.objects) == len(detected.objects),
            "objects_ground_truth": len(truth.objects), "objects_detected": len(detected.objects),
            "objects_matched": len(matches), "attributes_matched": attribute_matches,
            "relations_true": len(true_rel), "relations_detected": len(detected_rel),
            "relations_matched": len(mapped & detected_rel)}


def evaluate(output="outputs/evaluation", episodes=3, seed=42, neural=False, train_episodes=160, attributes_model=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generator = SceneGenerator(seed)
    rng = np.random.default_rng(seed)
    encoder, losses = None, []
    attribute_encoder = None
    if attributes_model:
        from src.vision.attribute_cnn import load_attribute_model
        attribute_encoder = load_attribute_model(attributes_model)
    neural_status = "not requested; optional TensorFlow dependency and training required"
    if neural:
        from src.baselines.tensorflow_prototypical import train_encoder
        encoder, losses = train_encoder(targets()[:3], train_episodes, seed)
        print("Saving the trained encoder…", flush=True)
        encoder.save(output / "encoder.keras")
        print("Encoder saved; evaluating held-out problems…", flush=True)
        neural_status = f"fresh CNN trained on {train_episodes} episodes of the first three rule families; {encoder.count_params()} parameters"
    rows, perception, failures = [], [], []
    for target_index, target in enumerate(targets()):
        print(f"Evaluating target {target_index+1}/{len(targets())}: {verbalize(target)}", flush=True)
        for episode in range(episodes):
            samples = generator.balanced_examples(target, 6)
            # Exactly one panel per class withheld from the 12-panel problem.
            hidden_indices = (5, 11)
            support = [example for i, example in enumerate(samples) if i not in hidden_indices]
            hidden = [samples[i] for i in hidden_indices]
            if episode == 0:
                draw_page(([s for s, y in samples if y], [s for s, y in samples if not y])).save(output / f"problem_{target_index}.png")
            result = search(support)
            inferred = result.ranked[0].rule
            semantic_test = generator.pool(100)
            semantic_match = all(inferred.evaluate(s) == target.evaluate(s) for s in semantic_test)
            generated_valid = []
            for label in (True, False):
                try: generated_valid.append(inferred.evaluate(generator.sample(inferred, label)) == label)
                except GenerationError as error: failures.append({"rule": verbalize(inferred), "error": str(error)})
            base = {"target": verbalize(target), "target_index": target_index, "episode": episode,
                    "depth": target.depth, "exact_program_recovery": inferred == target,
                    "semantic_recovery_100_probes": semantic_match,
                    "generated_validity": sum(generated_valid)/2, "search_seconds": result.seconds,
                    "perfect_hypotheses": result.perfect_count}
            rows.append(dict(base, system="symbolic_ground_truth", condition="clean",
                             accuracy=float(np.mean([inferred.evaluate(s) == y for s, y in hidden])),
                             train_consistency=result.ranked[0].accuracy, explanation=verbalize(inferred)))
            if rows[-1]["accuracy"] < 1 or not semantic_match:
                failures.append({"target": verbalize(target), "episode": episode,
                                 "system": "symbolic_ground_truth", "condition": "clean",
                                 "error": "Support-fitting program disagrees with target on held-out panels or semantic probes",
                                 "explanation": verbalize(inferred), "held_out_accuracy": rows[-1]["accuracy"],
                                 "semantic_probe_match": semantic_match})
            for condition in ("clean", "rotation", "noise"):
                rendered_support = [render_condition(s, condition, rng) for s, _ in support]
                rendered_hidden = [render_condition(s, condition, rng) for s, _ in hidden]
                support_images = [image for image, _ in rendered_support]
                hidden_images = [image for image, _ in rendered_hidden]
                perception_truth = [truth for _, truth in rendered_support+rendered_hidden]
                predictions = prototype_predict(support_images, [y for _, y in support], hidden_images)
                rows.append(dict(target_index=target_index, episode=episode, system="pixel_prototype", condition=condition,
                                 accuracy=float(np.mean(predictions == [y for _, y in hidden]))))
                if encoder is not None:
                    predictions = prototype_predict(support_images, [y for _, y in support], hidden_images, encoder)
                    rows.append(dict(target_index=target_index, episode=episode, system="tensorflow_prototype", condition=condition,
                                     accuracy=float(np.mean(predictions == [y for _, y in hidden]))))
                extracted_support = [(extract_scene(np.array(im), representation="geometric"), y) for im, (_, y) in zip(support_images, support)]
                extracted_hidden = [(extract_scene(np.array(im), representation="geometric"), y) for im, (_, y) in zip(hidden_images, hidden)]
                cv_result = search(extracted_support)
                noisy_result = search(extracted_support, noisy=True, complexity_penalty=.015)
                for scene, (detected, _) in zip(perception_truth, extracted_support+extracted_hidden):
                    perception.append(dict(condition=condition, **_perception_metrics(scene, detected)))
                cv_rule = cv_result.ranked[0].rule
                rows.append(dict(target_index=target_index, episode=episode, system="symbolic_cv", condition=condition,
                                 accuracy=float(np.mean([cv_rule.evaluate(s) == y for s, y in extracted_hidden])),
                                 exact_program_recovery=cv_rule == target,
                                 semantic_recovery_100_probes=all(cv_rule.evaluate(s) == target.evaluate(s) for s in semantic_test),
                                 train_consistency=cv_result.ranked[0].accuracy,
                                 perfect_hypotheses=cv_result.perfect_count, explanation=verbalize(cv_rule)))
                noisy_rule = noisy_result.ranked[0].rule
                rows.append(dict(target_index=target_index, episode=episode, system="symbolic_cv_regularized", condition=condition,
                                 accuracy=float(np.mean([noisy_rule.evaluate(s) == y for s, y in extracted_hidden])),
                                 exact_program_recovery=noisy_rule == target,
                                 semantic_recovery_100_probes=all(noisy_rule.evaluate(s) == target.evaluate(s) for s in semantic_test),
                                 train_consistency=noisy_result.ranked[0].accuracy,
                                 perfect_hypotheses=noisy_result.perfect_count, explanation=verbalize(noisy_rule)))
                if cv_result.ranked[0].accuracy < 1:
                    failures.append({"target": verbalize(target), "condition": condition,
                                     "error": "No perfectly consistent CV rule", "best_accuracy": cv_result.ranked[0].accuracy})
                cv_hidden_accuracy = float(np.mean([cv_rule.evaluate(s) == y for s, y in extracted_hidden]))
                if cv_hidden_accuracy < 1:
                    failures.append({"target": verbalize(target), "episode": episode, "system": "symbolic_cv",
                                     "condition": condition, "error": "Held-out classification failed despite support fit",
                                     "explanation": verbalize(cv_rule), "accuracy": cv_hidden_accuracy})
                if attribute_encoder is not None:
                    neural_support = [(extract_scene(np.array(im), attribute_model=attribute_encoder, uncertain_only=False, representation="geometric"), y)
                                      for im, (_, y) in zip(support_images, support)]
                    neural_hidden = [(extract_scene(np.array(im), attribute_model=attribute_encoder, uncertain_only=False, representation="geometric"), y)
                                     for im, (_, y) in zip(hidden_images, hidden)]
                    neural_result = search(neural_support)
                    neural_rule = neural_result.ranked[0].rule
                    accuracy = float(np.mean([neural_rule.evaluate(s) == y for s, y in neural_hidden]))
                    rows.append(dict(target_index=target_index, episode=episode, system="neural_perception_hard_symbolic", condition=condition,
                                     accuracy=accuracy, train_consistency=neural_result.ranked[0].accuracy,
                                     exact_program_recovery=neural_rule == target,
                                     semantic_recovery_100_probes=all(neural_rule.evaluate(s) == target.evaluate(s) for s in semantic_test),
                                     perfect_hypotheses=neural_result.perfect_count, explanation=verbalize(neural_rule)))
                    for scene, (detected, _) in zip(perception_truth, neural_support+neural_hidden):
                        perception.append(dict(condition=condition, pipeline="neural", **_perception_metrics(scene, detected)))
                    if accuracy < 1:
                        failures.append({"target": verbalize(target), "episode": episode,
                                         "system": "neural_perception_hard_symbolic", "condition": condition,
                                         "error": "Neural attributes plus hard reasoning misclassified a held-out panel",
                                         "accuracy": accuracy, "explanation": verbalize(neural_rule)})
    # Reuse the identical scenes, rules and prediction matrix for both strategies.
    active = []
    for episode in range(episodes):
        prototype = AkinatorSession(seed=seed+episode, pool_size=220)
        for target in targets():
            for strategy in ("active", "random"):
                session = AkinatorSession.__new__(AkinatorSession)
                session.rules, session.scenes, session.predictions = prototype.rules, prototype.scenes, prototype.predictions
                session.survivors = np.arange(len(session.rules))
                session.history, session.strategy = [], strategy
                session.rule_weights = prototype.rule_weights
                session.rng = np.random.default_rng(seed+episode)
                session.current = session._next()
                start = perf_counter()
                while session.current is not None and len(session.history) < 40:
                    session.answer(session.evaluate_query(target))
                active.append({"target": verbalize(target), "episode": episode, "strategy": strategy,
                               "questions": len(session.history), "survivors": len(session.survivors),
                               "drawings_seen": sum(len(a.query_indices) or 1 for a in session.history),
                               "remaining_pool_behaviors": session.equivalence_count,
                               "target_survives": target in session.candidates,
                               "question_cap_reached": len(session.history) >= 40 and session.current is not None,
                               "seconds": perf_counter()-start,
                               "survivor_trace": [len(session.rules)]+[a.survivors for a in session.history]})
    aggregate = {}
    for system, condition in sorted({(r["system"], r["condition"]) for r in rows}):
        subset = [r for r in rows if r["system"] == system and r["condition"] == condition]
        aggregate[f"{system}/{condition}"] = {"held_out_accuracy": float(np.mean([r["accuracy"] for r in subset])), "problems": len(subset)}
        for metric in ("exact_program_recovery", "semantic_recovery_100_probes", "generated_validity", "search_seconds"):
            values = [r[metric] for r in subset if metric in r]
            if values: aggregate[f"{system}/{condition}"][metric] = float(np.mean(values))
    perception_summary = {}
    for pipeline, condition in sorted({(r.get("pipeline", "opencv"), r["condition"]) for r in perception}):
        subset = [r for r in perception if r.get("pipeline", "opencv") == pipeline and r["condition"] == condition]
        def ratio(numerator, denominator):
            return sum(r[numerator] for r in subset)/max(1, sum(r[denominator] for r in subset))
        perception_summary[f"{pipeline}/{condition}"] = {
            "object_count_accuracy": float(np.mean([r["object_count_correct"] for r in subset])),
            "object_precision": ratio("objects_matched", "objects_detected"),
            "object_recall": ratio("objects_matched", "objects_ground_truth"),
            "shape_and_fill_recall": ratio("attributes_matched", "objects_ground_truth"),
            "relation_precision": ratio("relations_matched", "relations_detected"),
            "relation_recall": ratio("relations_matched", "relations_true")}
    report = {"seed": seed, "episodes_per_target": episodes, "held_out_per_problem": 2,
              "neural_status": neural_status, "neural_losses": losses,
              "regularized_complexity_penalty": .015,
              "neural_perception_model": str(attributes_model) if attributes_model else None,
              "neural_perception_policy": "classify every segmented object; argmax attributes; all distributions retained" if attributes_model else None,
              "soft_symbolic_status": "planned; not implemented or measured",
              "rotation_condition": "Every object is rotated +18 degrees around its unchanged center; whole-page rotation is not used.",
              "active_query_protocol": "Active may use up to four independent ANY queries after six negatives. Random uses individual scenes. Compare drawings_seen alongside question counts; a question-count-only comparison is not equivalent labeling cost.",
              "perception_summary": perception_summary,
              "aggregate": aggregate, "problem_results": rows, "active_results": active,
              "perception_results": perception, "failures": failures,
              "real_hand_drawn_problems_tested": 0,
              "limitations": ["Small synthetic pilot; confidence intervals require a larger sample.",
                              "Semantic recovery means agreement on 100 probes, not logical equivalence.",
                              "Real hand-drawn evaluation has not been performed.",
                              "Training data may admit alternative rules; perfect_hypotheses records ambiguity.",
                              "Active and random stop at one behavior on a finite query pool, or 40 questions."]}
    (output/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = sorted({k for row in rows for k in row})
    with (output/"results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Measured synthetic pilot", "", f"Seed: {seed}. {episodes} problems per target, six target programs. Two panels held out per problem.", "", "| System / condition | Held-out accuracy |", "|---|---:|"]
    lines += [f"| {k} | {v['held_out_accuracy']:.1%} |" for k, v in aggregate.items()]
    lines += ["", f"Neural baseline: {neural_status}.", ""]
    lines += ["Rotation means +18° per object around its unchanged center. Noise means independent Gaussian RGB noise with standard deviation 18/255. They are separate conditions.", ""]
    if attributes_model:
        lines += [f"Object perception CNN: {attributes_model}. It receives isolated OpenCV regions, retains softmax distributions and supplies argmax attributes to the hard symbolic evaluator. Every segmented object is classified in this benchmark.", "", "Soft-symbolic evaluation is planned and has not been measured.", ""]
    for strategy in ("active", "random"):
        subset = [r for r in active if r["strategy"] == strategy]
        lines.append(f"{strategy.title()} questions: {np.mean([r['questions'] for r in subset]):.2f} mean; capped runs: {sum(r['question_cap_reached'] for r in subset)}.")
    lines += ["", "## Program recovery on clean problems", "", "| System | Exact AST | Agreement on 100 semantic probes |", "|---|---:|---:|"]
    for system in ("symbolic_ground_truth", "symbolic_cv", "neural_perception_hard_symbolic"):
        summary = aggregate.get(f"{system}/clean")
        if summary:
            lines.append(f"| {system} | {summary['exact_program_recovery']:.1%} | {summary['semantic_recovery_100_probes']:.1%} |")
    lines += ["", "## Perception", "", "| Pipeline / condition | Count correct | Object recall | Shape + fill recall | Relation recall |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {key} | {v['object_count_accuracy']:.1%} | {v['object_recall']:.1%} | {v['shape_and_fill_recall']:.1%} | {v['relation_recall']:.1%} |" for key, v in perception_summary.items()]
    lines += ["", "The regularized CV rows use accuracy − 0.015 × AST complexity; exact rows rank by consistency first. Simple and composed target results are recorded separately by target_index and depth in results.json."]
    lines += ["", "## Limits", ""]+[f"- {item}" for item in report["limitations"]]
    lines += ["", "## Observed failures", "", f"{len(failures)} failures recorded; inspect results.json for the exact programs and conditions."]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neural", action="store_true")
    parser.add_argument("--train-episodes", type=int, default=160)
    parser.add_argument("--attributes-model", default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.train_episodes < 1: parser.error("Episode counts must be positive")
    report = evaluate(args.output, args.episodes, args.seed, args.neural, args.train_episodes, args.attributes_model)
    print(json.dumps({"aggregate": report["aggregate"], "neural_status": report["neural_status"], "real_hand_drawn_problems_tested": 0}, indent=2))
