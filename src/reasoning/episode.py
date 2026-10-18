"""Few-shot classifiers fitted to positive AND negative visual supports.

No image/class/rule names enter fitting. Selection and standardization happen
inside support-only validation folds. Queries never change fitted parameters.
Scores are not probabilities, and an embedding classifier is not a recovered
symbolic rule. Abstention below is an empirical diagnostic, not a conformal or
distribution-free correctness guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
from itertools import product
from typing import Any
import numpy as np
from .features import FeatureBatch,EpisodeSupport


def _unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def _distances(left,right,cosine):
    if cosine: return np.clip(1-left@right.T,0,2)
    return np.sqrt(np.maximum(0,np.sum(left*left,axis=1)[:,None]+np.sum(right*right,axis=1)[None,:]-2*left@right.T))


def _balanced_accuracy(labels,predictions):
    return float(np.mean([np.mean(predictions[labels==label]==label) for label in (False,True)]))


@dataclass(frozen=True)
class Candidate:
    view: str = "global"
    normalization: str = "cosine"
    learner: str = "prototype"
    regularization: float = 1.
    top_k: int = 16

    def __post_init__(self):
        if self.normalization not in ("cosine","support_standardize","support_contrast") or self.learner not in ("prototype","ridge","rbf"):
            raise ValueError("Unsupported episode learner")
        if not self.view or self.regularization<=0: raise ValueError("Invalid episode learner parameters")
        if not isinstance(self.top_k,int) or self.top_k<1: raise ValueError("Feature budget must be positive")

    @property
    def key(self):
        return (self.view,self.normalization,self.learner,self.regularization,self.top_k)


@dataclass
class _Model:
    candidate: Candidate
    mean: np.ndarray
    deviation: np.ndarray
    origin: np.ndarray
    train: np.ndarray
    coefficients: np.ndarray
    intercept: float
    bandwidth: float
    score_scale: float

    def transform(self,x):
        values=(x-self.mean)/self.deviation
        return _unit(values) if self.candidate.normalization=="cosine" else values

    def scores(self,x):
        return self.scores_from_transformed(self.transform(x))

    def scores_from_transformed(self,z):
        if self.candidate.learner=="prototype": result=z@self.coefficients+self.intercept
        elif self.candidate.learner=="ridge": result=(z-self.origin)@self.train.T@self.coefficients
        else:
            distances=np.maximum(0,np.sum(z*z,axis=1)[:,None]+np.sum(self.train*self.train,axis=1)[None,:]-2*z@self.train.T)
            result=np.exp(-distances/self.bandwidth)@self.coefficients+self.intercept
        return result/max(self.score_scale,1e-8)


def _prepare(x,labels,normalization,top_k=16):
    if normalization in ("support_standardize","support_contrast"):
        mean=x.mean(axis=0)
        deviation=x.std(axis=0)
        active_count=max(1,int(sum(deviation>1e-10)))
        active=deviation[deviation>1e-10]
        floor=max(1e-8,float(np.median(active))*.05) if len(active) else 1.
        deviation=np.maximum(deviation,floor)
        if normalization=="support_contrast":
            standardized=(x-mean)/deviation
            positive,negative=standardized[labels],standardized[~labels]
            difference=np.abs(positive.mean(axis=0)-negative.mean(axis=0))
            within=.5*(positive.var(axis=0)+negative.var(axis=0))
            effect=difference/np.sqrt(within+.25)
            chosen=np.argsort(-effect,kind="stable")[:min(top_k,len(effect))]
            weight=np.zeros(len(effect))
            weight[chosen]=np.sqrt(effect[chosen]**2/(1+effect[chosen]**2))
            deviation=deviation/np.maximum(weight,1e-12)
            active_count=max(1,int(sum(weight>1e-8)))
        # Scale by dimensions, never per row: Euclidean measurements retain
        # magnitude (unit-normalizing a scalar would erase all count values).
        deviation=deviation*active_count**.5
    else: mean=np.zeros(x.shape[1]);deviation=np.ones(x.shape[1])
    z=(x-mean)/deviation
    if normalization=="cosine": z=_unit(z)
    y=np.where(labels,1.,-1.)
    weights=np.array([len(y)/(2*sum(labels==label)) for label in labels])
    origin=np.average(z,axis=0,weights=weights)
    train=(z-origin)*np.sqrt(weights)[:,None]
    distances=np.maximum(0,np.sum(z*z,axis=1)[:,None]+np.sum(z*z,axis=1)[None,:]-2*z@z.T)
    nonzero=distances[distances>1e-10]
    bandwidth=max(1e-6,float(np.median(nonzero))) if len(nonzero) else 1.
    return mean,deviation,z,y,weights,origin,train,train@train.T,np.exp(-distances/bandwidth),bandwidth


def _fit_model(x,labels,candidate,prepared=None):
    mean,deviation,z,y,weights,origin,train,linear_kernel,rbf_kernel,bandwidth=prepared or _prepare(x,labels,candidate.normalization,candidate.top_k)
    model=_Model(candidate,mean,deviation,origin,z,np.zeros(len(y)),0.,1.,1.)
    if candidate.learner=="prototype":
        positive,negative=z[labels].mean(axis=0),z[~labels].mean(axis=0)
        if candidate.normalization=="cosine": model.coefficients=_unit(positive[None,:])[0]-_unit(negative[None,:])[0]
        else:
            model.coefficients=positive-negative
            model.intercept=float((negative@negative-positive@positive)/2)
    elif candidate.learner=="ridge":
        model.train=train
        model.coefficients=np.linalg.solve(linear_kernel+candidate.regularization*np.eye(len(y)),np.sqrt(weights)*y)
    else:
        model.bandwidth=bandwidth
        # Weighted loss keeps group inversion symmetric with unequal supports.
        model.coefficients=np.linalg.solve(rbf_kernel+candidate.regularization*np.diag(1/weights),y)
    model.score_scale=max(.05,float(np.median(np.abs(model.scores_from_transformed(z)))))
    return model


def _candidates(views,profile="compact"):
    result=[]
    for view in sorted(views,key=lambda value:(value!="global",value)):
        result.extend((Candidate(view,"cosine","prototype"),Candidate(view,"cosine","ridge",1.),
                       Candidate(view,"cosine","rbf",.01),Candidate(view,"support_standardize","prototype"),
                       Candidate(view,"support_standardize","ridge",.01),Candidate(view,"support_standardize","rbf",.01),
                       Candidate(view,"support_contrast","prototype"),
                       Candidate(view,"support_contrast","ridge",1.)))
        if profile=="thorough":
            result.extend((Candidate(view,"cosine","rbf",1.),Candidate(view,"support_standardize","ridge",1.),
                           Candidate(view,"support_standardize","ridge",100.),Candidate(view,"support_contrast","ridge",.01)))
    return tuple(result)


def _folds(labels):
    if min(sum(labels),sum(~labels))<2: return []
    # Hold one positive and one negative out together: no validation group
    # receives an easier class prior, including the two-per-group case.
    return [np.array([a,b]) for a,b in product(np.flatnonzero(~labels),np.flatnonzero(labels))]


def _select(views,labels,candidates):
    folds=_folds(labels)
    if not folds: return candidates[0],[]
    scored=[];cache={}
    for index,candidate in enumerate(candidates):
        predictions=np.zeros(len(labels));counts=np.zeros(len(labels))
        for fold_index,held in enumerate(folds):
            keep=np.ones(len(labels),bool);keep[held]=False
            key=candidate.view,candidate.normalization,candidate.top_k,fold_index
            if key not in cache:
                prepared=_prepare(views[candidate.view][keep],labels[keep],candidate.normalization,candidate.top_k)
                mean,deviation=prepared[:2]
                validation=(views[candidate.view][held]-mean)/deviation
                if candidate.normalization=="cosine": validation=_unit(validation)
                cache[key]=prepared,validation
            prepared,validation=cache[key]
            model=_fit_model(views[candidate.view][keep],labels[keep],candidate,prepared)
            predictions[held]+=model.scores_from_transformed(validation);counts[held]+=1
        predictions/=np.maximum(counts,1)
        accuracy=_balanced_accuracy(labels,predictions>0)
        target=np.where(labels,1.,-1.)
        loss=float(np.mean(np.logaddexp(0,-np.clip(predictions,-20,20)*target)))
        scored.append((candidate,accuracy,loss,index))
    # Rounding prevents floating point summation order from changing a tie.
    scored.sort(key=lambda item:(-round(item[1],10),round(item[2],10),item[3]))
    return scored[0][0],[{"candidate":asdict(c),"accuracy":a,"log_loss":loss} for c,a,loss,_ in scored]


@dataclass(frozen=True)
class AbstentionPolicy:
    minimum_cv_accuracy: float = .65
    minimum_margin: float = .15
    minimum_agreement: float = .8
    novelty_factor: float = 1.5
    novelty_floor: float = .08

    def __post_init__(self):
        if not .5<=self.minimum_cv_accuracy<=1 or not .5<=self.minimum_agreement<=1:
            raise ValueError("Invalid validation or agreement threshold")
        if self.minimum_margin<0 or self.novelty_factor<1 or self.novelty_floor<0:
            raise ValueError("Invalid margin or novelty threshold")


@dataclass(frozen=True)
class EpisodePrediction:
    image_id: str
    label: bool | None
    forced_label: bool | None
    score: float
    probability: float | None
    abstention_reasons: tuple[str,...]
    nearest_support_ids: tuple[str,...]
    diagnostics: dict[str,Any]

    def to_dict(self): return asdict(self)


@dataclass
class FittedEpisode:
    support: EpisodeSupport
    model: _Model
    validation_models: tuple[_Model,...]
    diagnostics: dict[str,Any]
    policy: AbstentionPolicy
    novelty_cutoff: float

    def predict(self,query:FeatureBatch):
        if query.encoder_id!=self.support.features.encoder_id: raise ValueError("Query and support encoders must match exactly")
        if set(query.image_ids)&set(self.support.features.image_ids):
            raise ValueError("Query ids overlap support ids; use held-out support validation instead")
        views=query.views()
        needed={self.model.candidate.view,*(m.candidate.view for m in self.validation_models)}
        support_views=self.support.features.views()
        if any(name not in views or views[name].shape[1]!=support_views[name].shape[1] for name in needed):
            raise ValueError("Query is missing a fitted feature view or has different feature dimensions")
        scores=self.model.scores(views[self.model.candidate.view])
        validation=np.stack([model.scores(views[model.candidate.view]) for model in self.validation_models]) if self.validation_models else np.zeros((0,len(scores)))
        if query.global_features.shape[1]!=self.support.features.global_features.shape[1]: raise ValueError("Global query dimensions differ from support")
        reference=self.model.transform(support_views[self.model.candidate.view])
        cosine=self.model.candidate.normalization=="cosine"
        distances=_distances(self.model.transform(views[self.model.candidate.view]),reference,cosine)
        output=[]
        for index,(image_id,score) in enumerate(zip(query.image_ids,scores)):
            forced=None if abs(score)<=1e-10 else bool(score>0)
            agreement=float(np.mean((validation[:,index]>0)==forced)) if len(validation) and forced is not None else 0.
            reasons=[]
            if self.diagnostics["contradictory_pairs"]: reasons.append("contradictory_support")
            cv=self.diagnostics["nested_cv_accuracy"]
            if cv is None: reasons.append("insufficient_support_for_validation")
            elif cv<self.policy.minimum_cv_accuracy: reasons.append("weak_support_validation")
            if forced is None or abs(score)<self.policy.minimum_margin: reasons.append("small_decision_margin")
            if agreement<self.policy.minimum_agreement: reasons.append("unstable_support_fit")
            nearest=float(distances[index].min())
            if nearest>self.novelty_cutoff: reasons.append("outside_support_neighborhood")
            ranking=np.argsort(distances[index],kind="stable")[:min(3,len(reference))]
            output.append(EpisodePrediction(image_id,None if reasons else forced,forced,float(score),None,tuple(reasons),
                tuple(self.support.features.image_ids[i] for i in ranking),
                {"fit_agreement":agreement,"nearest_support_distance":nearest,
                 "nearest_cosine_distance":nearest if cosine else None,"novelty_cutoff":self.novelty_cutoff,
                 "distance_metric":"cosine" if cosine else "support_standardized_euclidean",
                 "distance_feature_view":self.model.candidate.view,
                 "support_cv_accuracy":cv,"score_is_probability":False,"rule_recovered":False}))
        return output


class EpisodeClassifier:
    def __init__(self,candidates=None,policy=None,search_profile="compact"):
        if search_profile not in ("compact","thorough"): raise ValueError("Unknown episode selection profile")
        self.candidates=None if candidates is None else tuple(candidates)
        self.policy=policy or AbstentionPolicy()
        self.search_profile=search_profile

    def fit(self,support:EpisodeSupport|FeatureBatch,labels=None):
        if isinstance(support,FeatureBatch): support=EpisodeSupport(support,tuple(labels) if labels is not None else ())
        elif labels is not None: raise ValueError("Pass labels either in EpisodeSupport or as a separate argument")
        order=np.argsort(support.features.image_ids,kind="stable")
        support=EpisodeSupport(support.features.subset(order),tuple(support.labels[i] for i in order),support.episode_id)
        views=support.features.views();y=np.array(support.labels,bool)
        automatic=support.features.metadata.get("automatic_feature_views",tuple(views))
        if any(name not in views for name in automatic): raise ValueError("Unknown automatic feature view")
        allowed={name:views[name] for name in ("global",*automatic)}
        candidates=self.candidates or _candidates(allowed,self.search_profile)
        if not candidates or any(candidate.view not in views for candidate in candidates): raise ValueError("Unavailable candidate feature view")
        selected,selection=_select(views,y,candidates)
        model=_fit_model(views[selected.view],y,selected)
        outer_models=[];outer_scores=np.full(len(y),np.nan)
        # Outer LOO re-selects the learner using only its remaining supports.
        # Chosen-model CV on all supports is reported separately as optimistic.
        if min(sum(y),sum(~y))>=2:
            for held in range(len(y)):
                keep=np.arange(len(y))!=held
                inner_views={name:value[keep] for name,value in views.items()}
                candidate,_=_select(inner_views,y[keep],candidates)
                partial=_fit_model(inner_views[candidate.view],y[keep],candidate)
                outer_scores[held]=partial.scores(views[candidate.view][held:held+1])[0]
                outer_models.append(partial)
        cv=_balanced_accuracy(y,outer_scores>0) if np.isfinite(outer_scores).all() else None
        contradictory=[]
        for a in range(len(y)):
            for b in range(a):
                if y[a]!=y[b] and all(np.allclose(value[a],value[b],rtol=1e-8,atol=1e-10) for value in views.values()):
                    contradictory.append((support.features.image_ids[b],support.features.image_ids[a]))
        reference=model.transform(views[selected.view])
        cosine=model.candidate.normalization=="cosine"
        distances=_distances(reference,reference,cosine);np.fill_diagonal(distances,np.inf)
        nearest=distances.min(axis=1)
        floor=self.policy.novelty_floor if cosine else (2*self.policy.novelty_floor)**.5
        novelty=max(floor,float(np.max(nearest))*self.policy.novelty_factor)
        diagnostics={"selected_candidate":asdict(selected),"support_counts":{"A":int(sum(y)),"B":int(sum(~y))},
            "nested_cv_accuracy":cv,"nested_cv_scores":[None if not np.isfinite(s) else float(s) for s in outer_scores],
            "nested_cv_predictions":[None if not np.isfinite(s) or abs(s)<=1e-10 else bool(s>0) for s in outer_scores],
            "selection_cv":selection,"contradictory_pairs":contradictory,"novelty_cutoff":novelty,
            "calibration":"Uncalibrated scores. Nested support validation is an empirical diagnostic, not a guarantee.",
            "automatic_feature_views":list(allowed),"candidate_count":len(candidates),
            "selection_profile":self.search_profile,"validation_protocol":"Outer leave-one-out; inner leave-one-per-group-out",
            "abstention_policy":asdict(self.policy),
            "executable_rule_recovered":False,"query_labels_used":False}
        return FittedEpisode(support,model,tuple(outer_models),diagnostics,self.policy,novelty)


def nearest_prototype(support:EpisodeSupport,query:FeatureBatch):
    """Fixed global cosine prototype baseline; no learner/view selection."""
    model=_fit_model(support.features.global_features,np.array(support.labels,bool),Candidate())
    if support.features.encoder_id!=query.encoder_id: raise ValueError("Encoder mismatch")
    return model.scores(query.global_features)
