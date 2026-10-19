"""Bounded language-search protocol, exercised without a pretrained model.

These scripted backends test control flow and information boundaries, not the
visual accuracy or statistical independence of language-model verification.
"""
import json
import numpy as np

from src.reasoning.concepts import ConceptInducer


def picture(value):
    return np.full((12,12,3),value,np.uint8)


def inputs():
    positive=[picture(20),picture(30)]
    negative=[picture(210),picture(220)]
    # A chromatic query marker cannot occur in a sheet of grayscale supports.
    query=np.tile(np.array([255,0,255],np.uint8),(12,12,1))
    return positive,negative,[query]


def hypothesis(name):
    return {"positive":name,"negative":"not "+name}


class ScriptedBackend:
    model_id="scripted-local-protocol-test"

    def __init__(self,rounds,*,query_votes=None):
        self.rounds=rounds
        self.query_votes=query_votes or {}
        self.calls=[]
        self.proposal_count=0

    def complete(self,prompt,images,*,max_new_tokens):
        copied=[np.array(image,copy=True) for image in images]
        if prompt.startswith("Solve this Bongard concept problem."):
            self.calls.append({"kind":"proposal","prompt":prompt,"images":copied})
            result=self.rounds[min(self.proposal_count,len(self.rounds)-1)]
            self.proposal_count+=1
            return json.dumps({"hypotheses":result})
        assert prompt.startswith("Independently examine this ONE image.")
        assert len(copied)==1
        image=copied[0]
        is_query=bool(np.any(image[...,0]!=image[...,1]))
        description=prompt.split("Category A: ",1)[1].split(". Category B:",1)[0].casefold().strip()
        if is_query:
            label=self.query_votes.get(description,"A")
            evidence="QUERY_SENTINEL: unseen image observation"
        elif description=="bad criterion":
            label="A"  # Deliberately fails every negative support.
            evidence="visible support observation"
        else:
            label="A" if image.mean()<128 else "B"
            evidence="visible support observation"
        self.calls.append({"kind":"query" if is_query else "support","prompt":prompt,
                           "images":copied,"description":description,"label":label})
        return json.dumps({"label":label,"evidence":evidence})


def test_failed_candidate_is_repaired_using_support_feedback_only():
    backend=ScriptedBackend([[hypothesis("bad criterion")],[hypothesis("good criterion")]])
    result=ConceptInducer(backend,max_candidates=1,max_rounds=3).solve(*inputs())
    assert [candidate.consistent for candidate in result.candidates]==[False,True]
    assert [candidate.support_correct for candidate in result.candidates]==[2,4]
    assert result.diagnostics["support_search_rounds"]==2
    assert result.diagnostics["consistent_concepts"]==1
    assert result.predictions[0].label is True
    proposals=[call for call in backend.calls if call["kind"]=="proposal"]
    assert len(proposals)==2
    assert "Feedback:" not in proposals[0]["prompt"]
    feedback=json.loads(proposals[1]["prompt"].split("Feedback: ",1)[1])
    assert feedback[0]["hypothesis"]["positive"]=="bad criterion"
    assert {item["image"] for item in feedback[0]["failed_supports"]}=={"B1","B2"}
    assert {item["observed_judgment"] for item in feedback[0]["failed_supports"]}=={True}
    assert all("QUERY_SENTINEL" not in call["prompt"] for call in proposals)
    assert len([call for call in backend.calls if call["kind"]=="support"])==8


def test_repeated_candidate_is_skipped_and_search_can_continue():
    repeated={"positive":"  BAD CRITERION  ","negative":"  NOT BAD CRITERION  "}
    backend=ScriptedBackend([[hypothesis("bad criterion")],[repeated],[hypothesis("good criterion")]])
    result=ConceptInducer(backend,max_candidates=1,max_rounds=3).solve(*inputs())
    assert backend.proposal_count==3
    assert len(result.candidates)==2
    assert result.diagnostics["support_search_rounds"]==3
    assert len([call for call in backend.calls if call.get("description")=="bad criterion"])==4
    assert result.predictions[0].label is True


def test_repetition_budget_terminates_without_ever_judging_queries():
    backend=ScriptedBackend([[hypothesis("bad criterion")]])
    result=ConceptInducer(backend,max_candidates=1,max_rounds=4).solve(*inputs())
    assert backend.proposal_count==4
    assert len(result.candidates)==1
    assert len([call for call in backend.calls if call["kind"]=="support"])==4
    assert not any(call["kind"]=="query" for call in backend.calls)
    assert result.predictions[0].label is None
    assert result.predictions[0].votes==()
    assert result.diagnostics["consistent_concepts"]==0


def test_proposal_and_repair_never_receive_query_pixels_or_observations():
    backend=ScriptedBackend([[hypothesis("bad criterion")],[hypothesis("good criterion")]])
    ConceptInducer(backend,max_candidates=1,max_rounds=2).solve(*inputs())
    first_query=next(index for index,call in enumerate(backend.calls) if call["kind"]=="query")
    assert max(index for index,call in enumerate(backend.calls) if call["kind"]=="proposal")<first_query
    for call in backend.calls:
        if call["kind"]=="proposal":
            assert len(call["images"])==1
            sheet=call["images"][0]
            assert sheet.shape[0]>12 and sheet.shape[1]>12
            assert np.array_equal(sheet[...,0],sheet[...,1])
            assert np.array_equal(sheet[...,1],sheet[...,2])
            assert "QUERY_SENTINEL" not in call["prompt"]
        else:
            assert len(call["images"])==1
            assert "upper row" not in call["prompt"]
            assert "lower row" not in call["prompt"]
            assert "Feedback:" not in call["prompt"]
            assert "A1" not in call["prompt"] and "B1" not in call["prompt"]


def test_support_consistent_hypotheses_that_conflict_on_query_abstain():
    backend=ScriptedBackend([[hypothesis("criterion one"),hypothesis("criterion two")]],
                            query_votes={"criterion one":"A","criterion two":"B"})
    result=ConceptInducer(backend,max_candidates=2,max_rounds=4).solve(*inputs())
    assert backend.proposal_count==1
    assert all(candidate.consistent for candidate in result.candidates)
    assert result.diagnostics["consistent_concepts"]==2
    assert result.predictions[0].votes==(True,False)
    assert result.predictions[0].label is None
    assert "disagree" in result.predictions[0].reason
    assert result.diagnostics["support_consistency_is_heldout_accuracy"] is False


def test_one_unresolved_query_verification_does_not_get_outvoted():
    backend=ScriptedBackend([[hypothesis("criterion one"),hypothesis("criterion two")]],
                            query_votes={"criterion one":"A","criterion two":"unknown"})
    result=ConceptInducer(backend,max_candidates=2).solve(*inputs())
    assert result.predictions[0].votes==(True,None)
    assert result.predictions[0].label is None
