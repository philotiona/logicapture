from streamlit.testing.v1 import AppTest
from pathlib import Path
import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_interface_navigation_and_solve():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    app.button(key="nav_bongard").click().run()
    next(r for r in app.radio if r.label == "Reasoning approach").set_value("Executable structural rules").run()
    assert not app.exception
    next(b for b in app.button if b.label == "Read drawings").click().run()
    assert "read_drawings" in app.session_state
    assert "solved" not in app.session_state
    next(b for b in app.button if b.label == "Infer rule").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state.solved["result"].ranked[0].accuracy == 1
    assert app.session_state.solved["result"].ranked[0].total == 12
    assert app.session_state.solved["hidden"] == []


def test_interface_akinator_answer():
    app = AppTest.from_file(APP, default_timeout=30).run()
    next(b for b in app.button if b.label == "Start session").click().run()
    assert not app.exception
    original = len(app.session_state.akinator.survivors)
    next(b for b in app.button if b.label == "Yes").click().run()
    assert not app.exception
    assert len(app.session_state.akinator.survivors) < original


def test_interface_neural_solver_retains_distributions():
    pytest.importorskip("tensorflow")
    if not Path("data/models/attributes.keras").exists(): pytest.skip("Object model not trained yet")
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key="nav_bongard").click().run()
    next(r for r in app.radio if r.label == "Reasoning approach").set_value("Executable structural rules").run()
    next(r for r in app.radio if r.label == "Perception pipeline").set_value("Neural perception + hard symbolic").run()
    next(b for b in app.button if b.label == "Read drawings").click().run()
    next(b for b in app.button if b.label == "Infer rule").click().run()
    assert not app.exception and not app.error
    objects = [obj for scene, _ in app.session_state.solved["scenes"] for obj in scene.objects]
    assert any(obj.shape_probabilities is not None and obj.fill_probabilities is not None for obj in objects)


def test_bongard_input_change_requires_new_reading():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key="nav_bongard").click().run()
    next(r for r in app.radio if r.label == "Reasoning approach").set_value("Executable structural rules").run()
    next(b for b in app.button if b.label == "Read drawings").click().run()
    next(b for b in app.button if b.label == "Infer rule").click().run()
    next(s for s in app.selectbox if s.label == "Example problem").select("Exactly two objects").run()
    assert not any(b.label == "Infer rule" for b in app.button)
    assert not app.exception


def test_bongard_optional_holdout():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key="nav_bongard").click().run()
    next(r for r in app.radio if r.label == "Reasoning approach").set_value("Executable structural rules").run()
    next(b for b in app.button if b.label == "Read drawings").click().run()
    next(c for c in app.checkbox if c.label == "Reserve one drawing per group for evaluation").check().run()
    next(b for b in app.button if b.label == "Infer rule").click().run()
    assert app.session_state.solved["result"].ranked[0].total == 10
    assert len(app.session_state.solved["hidden"]) == 2
    assert not app.exception


def test_supplied_arrow_problem_uses_structural_pixels_and_checked_examples():
    app = AppTest.from_file(APP, default_timeout=45).run()
    app.button(key="nav_bongard").click().run()
    next(r for r in app.radio if r.label == "Reasoning approach").set_value("Executable structural rules").run()
    pipeline = next(r for r in app.radio if r.label == "Perception pipeline")
    assert pipeline.value == "Structural graph + symbolic"
    next(s for s in app.selectbox if s.label == "Example problem").select("Arrow directions / supplied drawing").run()
    next(b for b in app.button if b.label == "Read drawings").click().run()
    assert not app.exception and not app.error
    scenes = app.session_state.read_drawings["scenes"]
    assert len(scenes) == 12
    for scene, label in scenes:
        assert len(scene.objects) == 1
        obj = scene.objects[0]
        assert obj.kind == "path" and obj.shape == "unknown"
        assert obj.path.arrow_relation == ("opposed" if label else "same")
    next(b for b in app.button if b.label == "Infer rule").click().run()
    result = app.session_state.solved["result"]
    assert result.ranked[0].correct == 12 and result.ranked[0].unknown == 0
    next(b for b in app.button if b.label == "Generate examples").click().run()
    rule = result.ranked[0].rule
    generated = app.session_state.examples_solver[1]
    assert rule.assess(generated[0]) is True and rule.assess(generated[1]) is False
    assert all(scene.objects[0].kind == "path" for scene in generated)
    assert not app.exception and not app.error
