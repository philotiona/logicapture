"""Held-out rendered graph problems; labels never enter perception.

This is a synthetic regression, not a claim about arbitrary handwritten Bongard
problems. It complements the separate supplied-drawing validation.
"""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np

from src.active.path_generation import PathSceneGenerator
from src.dsl.nodes import Exists,Predicate
from src.dsl.verbalizer import verbalize
from src.scene.renderer import draw_page,draw_scene
from src.synthesis.search import search
from src.vision.panel_detection import detect_panels
from src.vision.object_detection import extract_scene

CONCEPTS={"opposing_arrows":Predicate("arrow_relation","opposed"),
          "open_paths":Predicate("topology","open"),
          "straight_paths":Predicate("geometry","straight"),
          "arrow_presence":Predicate("has_arrows")}


def _metrics(predictions,labels):
    known=sum(value is not None for value in predictions)
    correct=sum(value is label for value,label in zip(predictions,labels))
    return {"correct":correct,"total":len(labels),"known":known,"unknown":len(labels)-known,
            "wrong":known-correct,"accuracy":correct/len(labels),"coverage":known/len(labels)}


def evaluate_path_generalization(training_seed=113,heldout_seeds=(7,31,91),per_group=6):
    styles=((240,2),(208,3),(320,4))
    report={"protocol":"Infer from training page pixels; freeze AST; evaluate new renderer seeds and canvas/pen styles from panel pixels.",
            "scope":"Synthetic legible open/closed paths. This does not measure handwritten generalization.",
            "training_seed":training_seed,"heldout_seeds":list(heldout_seeds),"concepts":[]}
    for name,predicate in CONCEPTS.items():
        target=Exists(predicate)
        training=PathSceneGenerator(seed=training_seed,topologies=("open","closed")).balanced_examples(target,per_group)
        page=draw_page(([scene for scene,label in training if label],[scene for scene,label in training if not label]))
        panels=detect_panels(np.array(page))
        examples=[(extract_scene(panel.image,adaptive=True),panel.label) for panel in panels]
        result=search(examples,paths=True)
        inferred=result.ranked[0].rule
        item={"name":name,"intended":asdict(target),"inferred":asdict(inferred),"description":verbalize(inferred),
              "training":_metrics([inferred.assess(scene) for scene,_ in examples],[label for _,label in examples]),
              "heldout":[],"failures":[]}
        all_target,all_inferred,all_labels=[],[],[]
        for seed_index,seed in enumerate(heldout_seeds):
            canvas,stroke=styles[seed_index%len(styles)]
            drawings=PathSceneGenerator(seed=seed,topologies=("open","closed")).balanced_examples(target,per_group)
            labels=[label for _,label in drawings]
            scenes=[extract_scene(np.array(draw_scene(scene,size=canvas,stroke=stroke)),adaptive=True) for scene,_ in drawings]
            target_values=[target.assess(scene) for scene in scenes]
            inferred_values=[inferred.assess(scene) for scene in scenes]
            item["heldout"].append({"seed":seed,"canvas":canvas,"stroke":stroke,
                                    "intended":_metrics(target_values,labels),"inferred":_metrics(inferred_values,labels)})
            for index,(scene,label,value) in enumerate(zip(scenes,labels,target_values)):
                if value is not label:
                    item["failures"].append({"seed":seed,"index":index,"label":label,"assessment":value,
                                             "evidence":target.explain(scene)})
            all_target.extend(target_values);all_inferred.extend(inferred_values);all_labels.extend(labels)
        item["aggregate_intended"]=_metrics(all_target,all_labels)
        item["aggregate_inferred"]=_metrics(all_inferred,all_labels)
        report["concepts"].append(item)
    return report


if __name__=="__main__":
    report=evaluate_path_generalization()
    destination=Path("data/regressions/path_generalization.json")
    destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps([{ "name":item["name"],"description":item["description"],
                       "training":item["training"],"heldout_intended":item["aggregate_intended"],
                       "heldout_inferred":item["aggregate_inferred"],"failed_cases":len(item["failures"])}
                     for item in report["concepts"]],indent=2))
