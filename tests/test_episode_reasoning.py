"""Learner correctness tests use latent vectors, not fabricated vision accuracy."""
import numpy as np
import pytest
from src.reasoning import (FeatureBatch,EpisodeSupport,EpisodeClassifier,Candidate,AbstentionPolicy,
                           nearest_prototype,evaluate_episode,prediction_metrics,fuse_feature_batches)


def batch(x,prefix="s",patches=None,extra=None):
    return FeatureBatch(tuple(f"{prefix}{i:03d}" for i in range(len(x))),np.asarray(x),patches,
                        encoder_id="test-encoder@fixed",extra_views=extra or {})


@pytest.fixture(scope="module")
def context_episode():
    rng=np.random.default_rng(8)
    labels=np.repeat([False,True],6)
    query_labels=np.repeat([False,True],40)
    support=np.column_stack((np.where(labels,.1,-.1),rng.normal(0,10,(12,4))))
    queries=np.column_stack((np.where(query_labels,.1,-.1),rng.normal(0,10,(80,4))))
    data=EpisodeSupport(batch(support),tuple(labels),"latent-context-test")
    query=batch(queries,"q")
    return data,query,query_labels,EpisodeClassifier().fit(data)


def test_support_conditioning_ignores_high_variance_nuisance(context_episode):
    support,query,labels,model=context_episode
    baseline=np.mean((nearest_prototype(support,query)>0)==labels)
    metrics=prediction_metrics(model.predict(query),labels)
    assert baseline<.7
    assert metrics["forced_accuracy"]>.95
    assert metrics["forced_accuracy"]>baseline+.25
    assert metrics["coverage"]>.8
    assert metrics["selective_accuracy"]>.95
    assert model.diagnostics["selected_candidate"]["normalization"]=="support_standardize"
    assert model.diagnostics["nested_cv_accuracy"]>.9


def test_label_inversion_and_support_order_invariance(context_episode):
    support,query,_,model=context_episode
    reversed_labels=EpisodeSupport(support.features,tuple(not value for value in support.labels))
    inverted=EpisodeClassifier().fit(reversed_labels)
    order=np.random.default_rng(55).permutation(len(support.labels))
    permuted=EpisodeClassifier().fit(EpisodeSupport(support.features.subset(order),tuple(support.labels[i] for i in order)))
    assert permuted.diagnostics["selected_candidate"]==model.diagnostics["selected_candidate"]
    assert inverted.diagnostics["selected_candidate"]==model.diagnostics["selected_candidate"]
    for a,b,c in zip(model.predict(query),inverted.predict(query),permuted.predict(query)):
        assert a.score==pytest.approx(-b.score,abs=1e-10)
        assert a.score==pytest.approx(c.score,abs=1e-10)
        assert a.label==(None if b.label is None else not b.label)
        assert a.label==c.label
        assert a.abstention_reasons==b.abstention_reasons==c.abstention_reasons


def test_query_batch_cannot_change_fitting_or_other_queries(context_episode):
    support,query,_,model=context_episode
    single=query.subset([0])
    extra=batch(np.full((1,5),1000.),"novel")
    combined=FeatureBatch.concatenate([single,extra])
    first=model.predict(single)[0];second=model.predict(combined)[0]
    assert first.score==pytest.approx(second.score,abs=1e-12)
    assert first.label==second.label
    assert model.diagnostics["query_labels_used"] is False
    assert all(prediction.probability is None for prediction in model.predict(query))
    assert all(not prediction.diagnostics["rule_recovered"] for prediction in model.predict(query))


def test_nonlinear_concept_is_learned_without_a_concept_dictionary():
    rng=np.random.default_rng(9)
    corners=np.array([[-1,-1],[1,1],[-1,1],[1,-1]],float)
    x=np.repeat(corners,3,axis=0)+rng.normal(0,.03,(12,2))
    labels=np.repeat([True,True,False,False],3)
    query=np.repeat(corners,6,axis=0)+rng.normal(0,.05,(24,2))
    truth=np.repeat([True,True,False,False],6)
    support=EpisodeSupport(batch(x),tuple(labels))
    model=EpisodeClassifier().fit(support)
    assert model.diagnostics["selected_candidate"]["learner"]=="rbf"
    assert prediction_metrics(model.predict(batch(query,"q")),truth)["forced_accuracy"]==1.
    assert np.mean((nearest_prototype(support,batch(query,"q"))>0)==truth)<=.6


def test_patch_spatial_view_resolves_identical_global_embeddings():
    labels=np.repeat([False,True],3)
    patches=[]
    for label in labels:
        patch=np.tile([1.,0.],(4,1))
        patch[0 if label else 3]=[0.,1.]
        patches.append(patch)
    support=EpisodeSupport(batch(np.ones((6,2)),patches=tuple(patches)),tuple(labels))
    model=EpisodeClassifier().fit(support)
    assert model.diagnostics["selected_candidate"]["view"]=="patch_spatial"
    assert not model.diagnostics["contradictory_pairs"]
    query=batch(np.ones((2,2)),"q",patches=(patches[0],patches[-1]))
    assert [p.label for p in model.predict(query)]==[False,True]


def test_contradictory_duplicates_and_zero_information_abstain():
    support=EpisodeSupport(batch(np.ones((4,3))),(False,False,True,True))
    model=EpisodeClassifier().fit(support)
    predictions=model.predict(batch(np.ones((2,3)),"q"))
    assert model.diagnostics["contradictory_pairs"]
    assert all(p.label is None and "contradictory_support" in p.abstention_reasons for p in predictions)
    assert all("small_decision_margin" in p.abstention_reasons for p in predictions)


def test_two_per_group_is_valid_and_one_per_group_is_unvalidated():
    x=np.array([[-2.,0],[-1.,.1],[1.,.1],[2.,0]])
    model=EpisodeClassifier().fit(batch(x),[False,False,True,True])
    assert model.diagnostics["nested_cv_accuracy"]==1.
    assert [p.label for p in model.predict(batch(np.array([[-1.5,0],[1.5,0]]),"q"))]==[False,True]
    limited=EpisodeClassifier().fit(batch(x[[0,3]]),[False,True])
    assert limited.diagnostics["nested_cv_accuracy"] is None
    assert "insufficient_support_for_validation" in limited.predict(batch(x[[1]],"q"))[0].abstention_reasons


def test_boundary_and_outside_support_queries_abstain():
    x=np.array([[-1.,-.01],[-1.,.01],[1.,-.01],[1.,.01]])
    model=EpisodeClassifier().fit(batch(x),[False,False,True,True])
    query=batch(np.array([[0.,0.],[0.,1.]]),"q")
    predictions=model.predict(query)
    assert all(p.label is None for p in predictions)
    assert "small_decision_margin" in predictions[0].abstention_reasons
    assert "outside_support_neighborhood" in predictions[1].abstention_reasons


def test_query_labels_are_evaluation_only_and_overlap_is_rejected(context_episode):
    support,query,labels,model=context_episode
    original=evaluate_episode(support,query,labels)
    swapped=evaluate_episode(support,query,~labels)
    assert [p["score"] for p in original["predictions"]]==pytest.approx([p["score"] for p in swapped["predictions"]])
    assert original["fit"]==swapped["fit"]
    assert original["classifier"]["forced_accuracy"]!=swapped["classifier"]["forced_accuracy"]
    with pytest.raises(ValueError,match="overlap"): model.predict(support.features)
    with pytest.raises(ValueError,match="overlap"): evaluate_episode(support,support.features,np.array(support.labels))


def test_feature_validation_subsetting_concatenation_and_fusion():
    first=batch(np.array([[1.,2.],[3.,4.]]),extra={"measurement":np.array([[1.],[0.]])})
    restored=FeatureBatch.concatenate([first.subset([0]),first.subset([1])])
    assert np.array_equal(restored.global_features,first.global_features)
    assert np.array_equal(restored.extra_views["measurement"],first.extra_views["measurement"])
    other=FeatureBatch(tuple(reversed(first.image_ids)),np.array([[0.,1.],[1.,0.]]),encoder_id="other")
    fused=fuse_feature_batches({"first":first,"other":other})
    assert fused.image_ids==first.image_ids
    assert np.array_equal(fused.extra_views["other:global"],[[1.,0.],[0.,1.]])
    assert "first:measurement" in fused.extra_views
    with pytest.raises(ValueError): first.global_features[0,0]=0
    with pytest.raises(ValueError,match="unique"): FeatureBatch(("a","a"),np.ones((2,3)))
    with pytest.raises(ValueError,match="finite"): FeatureBatch(("a",),np.array([[np.nan]]))
    with pytest.raises(ValueError): EpisodeSupport(first,(1,0))


def test_diagnostic_feature_views_are_not_automatic_candidates():
    values=np.ones((4,2));labels=(False,False,True,True)
    features=FeatureBatch(tuple(f"s{i}" for i in range(4)),values,encoder_id="test",
                          extra_views={"diagnostic":np.array([[-1.],[-1.],[1.],[1.]])},
                          metadata={"automatic_feature_views":["global"]})
    model=EpisodeClassifier().fit(EpisodeSupport(features,labels))
    assert model.diagnostics["automatic_feature_views"]==["global"]
    assert model.diagnostics["selected_candidate"]["view"]=="global"
    fused=fuse_feature_batches({"representation":features})
    assert "representation:diagnostic" in fused.extra_views
    assert "representation:diagnostic" not in fused.metadata["automatic_feature_views"]


def test_numeric_magnitude_is_preserved_for_a_novel_interval_concept():
    values=np.array([-.2,-.1,-.05,.05,.1,.2,-1.,-1.2,-1.4,1.,1.2,1.4])[:,None]
    labels=np.array([True]*6+[False]*6)
    support=EpisodeSupport(batch(values),tuple(labels))
    query=batch(np.array([-.15,.15,-.9,-1.3,.9,1.3])[:,None],"q")
    truth=np.array([True,True,False,False,False,False])
    model=EpisodeClassifier().fit(support)
    assert model.diagnostics["selected_candidate"]["normalization"]=="support_standardize"
    assert model.diagnostics["selected_candidate"]["learner"]=="rbf"
    assert [p.forced_label for p in model.predict(query)]==list(truth)
