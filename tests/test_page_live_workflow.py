from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.vision.panel_detection import split_problem_page
from src.vision.observations import observations
from src.vision.live import LivePageEngine, analyze_live_panels
from src.reasoning.general import GeneralVisualSolver
from src.reasoning.examples import generate_checked_examples
from src.reasoning.measurement_rules import MeasurementRule
from src.datasets.fetch_foundalis import numbered_problems, MediaLinks

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def page():
    return np.array(Image.open(ROOT/"data/fixtures/dotted_page.png").convert("RGB"))


def test_whole_uploaded_book_page_splits_in_reading_order(page):
    _,panels,layout = split_problem_page(page)
    assert layout["positive"] == layout["negative"] == 6
    assert len(panels) == 12
    for group in (panels[:6],panels[6:]):
        assert all(group[i].box[0] < group[i+1].box[0] for i in (0,2,4))
        assert all(group[i].box[1] < group[i+2].box[1] for i in (0,1,2,3))
    assert all(p.label for p in panels[:6])
    assert not any(p.label for p in panels[6:])
    # No center binding enters the crops. Empty/partial marks do not turn
    # into fabricated total counts; interior dot spacing remains measurable.
    assert all(observations(p.image)["dot_spacing_p90"] is not None for p in panels)


def test_nested_black_islands_are_not_counted_as_white_holes():
    image = np.full((160,160,3),255,np.uint8)
    cv2.rectangle(image,(25,25),(135,135),(0,0,0),3)
    cv2.circle(image,(80,80),15,(0,0,0),-1)
    assert observations(image)["hole_count"] == 1
    cv2.circle(image,(80,80),7,(255,255,255),-1)
    assert observations(image)["hole_count"] == 2


def test_photographed_page_rectifies_and_splits_without_manual_crops(page):
    h,w = page.shape[:2]
    target = np.float32(((95,65),(830,115),(785,625),(55,590)))
    transform = cv2.getPerspectiveTransform(np.float32(((0,0),(w-1,0),(w-1,h-1),(0,h-1))),target)
    photograph = cv2.warpPerspective(page,transform,(900,700),borderValue=(65,65,65))
    _,panels,layout = split_problem_page(photograph)
    assert len(panels) == 12
    assert sum(p.label for p in panels) == 6
    # A visible outer paper boundary is optional: individual observed panel
    # quadrilaterals are rectified even when ink breaks the outer-page edge.
    assert all(len(p.corners)==4 for p in panels)


def test_one_missing_border_is_repaired_only_from_eleven_aligned_frames(page):
    # The lower-right border is absent in a real Foundalis image. This same
    # geometry should be repaired without a filename-specific exception.
    erased = page.copy()
    x,y,w,h = 484,272,126,126
    erased[y:y+4,x:x+w] = 255
    erased[y+h-7:y+h,x:x+w] = 255
    erased[y:y+h,x:x+4] = 255
    erased[y:y+h,x+w-7:x+w] = 255
    _,panels,layout = split_problem_page(erased)
    assert len(panels)==12
    assert layout["inferred_panels"]==1
    assert sum(p.inferred for p in panels)==1
    with pytest.raises(ValueError):
        split_problem_page(np.full_like(page,255))


@pytest.mark.parametrize("inverted",[False,True])
def test_dot_proximity_rule_is_fitted_without_heldout_answers(page,inverted):
    _,panels,_ = split_problem_page(page)
    a,b = [p.image for p in panels if p.label],[p.image for p in panels if not p.label]
    positive,negative = (b,a) if inverted else (a,b)
    result = GeneralVisualSolver(structural_first=True).solve(positive[:-1],negative[:-1],[a[-1],b[-1]])
    assert result.diagnostics["selected_pipeline"] == "executable_measurement_rules"
    assert [p.label for p in result.predictions] == [not inverted,inverted]
    assert result.diagnostics["query_labels_used"] is False
    check = result.diagnostics["structural_attempt"]
    assert check["query_images_used_in_selection"] is False
    assert check["validation_is_independent_accuracy"] is False
    assert "dot-width units" in result.diagnostics["logical_explanation"]
    examples = generate_checked_examples(result.diagnostics,attempts=24)
    assert [e.label for e in examples] == [True,False]
    programs = check["candidates"]
    for example in examples:
        assert not any(np.array_equal(example.image,s) for s in a+b)
        values = observations(example.image)
        for item in programs:
            p = item["program"]
            assert MeasurementRule(p["feature"],p["threshold"],p["positive_below"]).assess(values) is example.label
        assert example.verification["pixel_reread"]


class ControlledExecutor:
    def __init__(self): self.jobs = []
    def submit(self,analyzer,panels):
        future = Future()
        self.jobs.append((future,analyzer,panels))
        return future
    def shutdown(self,**kwargs): pass


def test_live_engine_waits_for_stability_and_discards_late_results(page):
    executor = ControlledExecutor()
    engine = LivePageEngine(executor=executor,interval=.1,stable_samples=3)
    engine.process(page,now=0)
    engine.process(page,now=1)
    assert not executor.jobs
    engine.process(page,now=2)
    assert len(executor.jobs) == 1
    engine.process(np.full_like(page,255),now=3)
    executor.jobs[0][0].set_result({"message":"STALE RULE","examples":[]})
    engine.process(np.full_like(page,255),now=4)
    assert engine.snapshot()["result"] is None
    assert engine.snapshot()["panels"] == ()
    assert "STALE" not in engine.snapshot()["message"]
    engine.close()


def test_sparse_page_changes_invalidate_old_rule_even_with_most_pixels_white():
    a = np.zeros((12,48,48),bool)
    b = a.copy()
    a[:,5:7,5:7] = True
    b[:,30:32,30:32] = True
    assert np.mean(a != b) < .01
    assert LivePageEngine._changed(a,b)
    shifted = np.roll(a,1,axis=1)
    assert not LivePageEngine._changed(a,shifted)


def test_failed_camera_readings_retry_fresh_frames_with_a_finite_budget(page):
    executor = ControlledExecutor()
    engine = LivePageEngine(executor=executor,interval=.1,stable_samples=2)
    failed = {"message":"No executable rule","structural_attempt":{"accepted":False},"examples":[]}
    engine.process(page,now=0)
    engine.process(page,now=1)
    executor.jobs[0][0].set_result(failed)
    engine.process(page,now=1.2)
    assert len(executor.jobs)==1
    engine.process(page,now=3.1)
    assert len(executor.jobs)==2
    executor.jobs[1][0].set_result(failed)
    engine.process(page,now=3.2)
    engine.process(page,now=5.2)
    assert len(executor.jobs)==3
    executor.jobs[2][0].set_result(failed)
    engine.process(page,now=5.3)
    engine.process(page,now=20)
    assert len(executor.jobs)==3
    engine.close()


def test_live_processing_draws_a_pixel_checked_positive_and_negative(page):
    _,panels,_ = split_problem_page(page)
    result = analyze_live_panels(panels)
    assert result["structural_attempt"]["accepted"]
    assert len(result["examples"]) == 2
    assert all(e.verification["pixel_reread"] for e in result["examples"])
    executor = ControlledExecutor()
    engine = LivePageEngine(executor=executor,interval=.1,stable_samples=2)
    engine.process(page,now=0)
    engine.process(page,now=1)
    executor.jobs[0][0].set_result(result)
    overlay = engine.process(page,now=2)
    assert overlay.shape[0] > page.shape[0]
    assert np.any(overlay[page.shape[0]:] < 128)
    assert engine.snapshot()["result"] is result
    engine.close()


def test_foundalis_discovery_uses_numeric_links_and_never_follows_solution_pages():
    html = '<a href="bongard/p001.htm"><font>1</font></a><a href="p002.htm">2</a>' \
           '<a href="solutions.htm">Solutions</a><a href="https://other.test/p003.htm">3</a>' \
           '<a href="bongard/p001.htm">1</a>'
    records = numbered_problems(html)
    assert [r["number"] for r in records] == [1,2]
    assert all("solutions" not in r["source_url"] for r in records)


def test_default_fullpage_flow_and_dotted_sample_generate_checked_drawings(monkeypatch):
    from src.vision import pretrained
    monkeypatch.setattr(pretrained,"model_directory",lambda *args:None)
    app = AppTest.from_file(ROOT/"app.py",default_timeout=45).run()
    app.button(key="nav_bongard").click().run()
    source = next(r for r in app.radio if r.label == "Input source")
    assert source.value == "Page image"
    source.set_value("Supplied dotted page").run()
    app.button(key="general_solve").click().run()
    assert not app.exception and not app.error
    saved = app.session_state.general_analysis
    assert saved["features"].diagnostics["selected_pipeline"] == "executable_measurement_rules"
    assert len(saved["generated"]) == 2
    assert any("2/2 correct, 0 unresolved" in c.value for c in app.caption)
