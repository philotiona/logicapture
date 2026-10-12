"""Reproduce the counted-attribute failure and its neighboring concepts."""
from dataclasses import asdict
import json
from pathlib import Path

from src.active import AkinatorSession, SceneGenerator
from src.dsl.nodes import Predicate, Count, ForAll, And
from src.dsl.verbalizer import verbalize


def evaluate(output="data/regressions/akinator_counts.json"):
    predicate = Predicate("and", children=(Predicate("shape", "triangle"), Predicate("filled")))
    targets = [Count(4, predicate), Count(4, predicate, True), And(Count(4), ForAll(predicate))]
    records = []
    for seed in (1, 7, 42, 121):
        for target in targets:
            session = AkinatorSession(seed=seed)
            while session.current is not None and len(session.history) < 30:
                session.answer(session.evaluate_query(target))
            recovered = session.candidates == [target]
            record = {"seed": seed, "target": verbalize(target), "program": asdict(target),
                      "questions": len(session.history), "drawings_seen": sum(len(a.query_indices) or 1 for a in session.history),
                      "batch_questions": sum(len(a.query_indices) > 1 for a in session.history),
                      "exact_recovery": recovered, "remaining": len(session.candidates),
                      "first_positive_question": next((i+1 for i, a in enumerate(session.history) if a.label), None),
                      "trace": [a.survivors for a in session.history]}
            records.append(record)
            print(f"seed {seed}: {record['target']} / {record['questions']} questions / recovered {recovered}", flush=True)
    report = {"scope": "Four seeds × three interpretations of four filled triangles; oracle answers, not a human study.",
              "protocol": "Single-drawing queries; after six negative answers, up to four independent drawings may be queried with ANY semantics. Both question and drawing counts are recorded.",
              "recovered": sum(r["exact_recovery"] for r in records), "runs": len(records), "results": records}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["recovered"] != len(records): raise RuntimeError("A counted-attribute regression failed")
    return report


if __name__ == "__main__": evaluate()
