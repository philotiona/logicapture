"""Query labels enter evaluation only, never episode fitting or selection."""
from dataclasses import dataclass,asdict
import numpy as np
from .features import EpisodeSupport,FeatureBatch
from .episode import EpisodeClassifier,nearest_prototype


def prediction_metrics(predictions,labels):
    labels=np.asarray(labels)
    if labels.dtype!=bool or len(labels)!=len(predictions): raise ValueError("Provide aligned boolean query labels")
    forced=np.array([prediction.forced_label for prediction in predictions],object)
    accepted=np.array([prediction.label is not None for prediction in predictions],bool)
    correct=np.array([prediction.label is bool(label) for prediction,label in zip(predictions,labels)],bool)
    forced_credit=np.array([.5 if value is None else float(value==label) for value,label in zip(forced,labels)])
    groups=[float(np.mean(forced_credit[labels==label])) for label in (False,True) if np.any(labels==label)]
    return {"total":len(labels),"answered":int(sum(accepted)),"abstained":int(sum(~accepted)),
            "coverage":float(np.mean(accepted)) if len(labels) else 0.,
            "selective_accuracy":float(np.mean(correct[accepted])) if np.any(accepted) else None,
            "accuracy_including_abstentions":float(np.mean(correct)) if len(labels) else None,
            "forced_accuracy":float(np.mean(forced_credit)) if len(labels) else None,
            "forced_balanced_accuracy":float(np.mean(groups)) if groups else None,
            "rule_recovery_measured":False,"probability_calibration_measured":False}


def evaluate_episode(support:EpisodeSupport,queries:FeatureBatch,query_labels,classifier=None):
    if set(support.features.image_ids)&set(queries.image_ids): raise ValueError("Evaluation query/support overlap")
    fitted=(classifier or EpisodeClassifier()).fit(support)
    predictions=fitted.predict(queries)
    baseline=nearest_prototype(support,queries)
    labels=np.asarray(query_labels)
    if labels.dtype!=bool or len(labels)!=len(queries.image_ids): raise ValueError("Aligned query labels required")
    baseline_credit=np.where(np.abs(baseline)<=1e-10,.5,(baseline>0)==labels)
    return {"episode_id":support.episode_id,"encoder_id":support.features.encoder_id,
            "classifier":prediction_metrics(predictions,labels),
            "nearest_prototype_forced_accuracy":float(np.mean(baseline_credit)),
            "predictions":[prediction.to_dict() for prediction in predictions],"fit":fitted.diagnostics}
