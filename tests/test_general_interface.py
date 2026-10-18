from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from src.reasoning.concepts import ConceptInducer
from src.reasoning.episode import EpisodePrediction
from src.reasoning.general import GeneralVisualSolver
from src.interface import general_bongard as interface
from src.vision import pretrained


def test_optional_language_failure_preserves_visual_predictions_and_releases_model(monkeypatch):
    released = []
    backend = SimpleNamespace(unload=lambda: released.append(True))
    monkeypatch.setattr(interface, "language_backend", lambda *args: backend)
    monkeypatch.setattr(interface, "semantic_encoder", lambda: SimpleNamespace(unload=lambda: None))
    monkeypatch.setattr(pretrained, "model_directory", lambda *args: Path("."))
    predictions = [EpisodePrediction(f"Q{i+1:02d}", None, False, 0., None, ("weak_support_validation",), ("A01",), {}) for i in range(2)]
    result = SimpleNamespace(predictions=predictions, diagnostics={"nested_cv_accuracy": .5,
                             "selected_candidate": {"view": "global", "learner": "prototype"}},
                             to_dict=lambda: {"visual_result_retained": True})
    monkeypatch.setattr(GeneralVisualSolver, "solve", lambda *args, **kwargs: result)
    def fail(*args, **kwargs): raise ValueError("Invalid concept JSON")
    monkeypatch.setattr(ConceptInducer, "solve", fail)
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=30).run()
    app.button(key="nav_bongard").click().run()
    assert next(r for r in app.radio if r.label == "Reasoning approach").value == "General visual inference"
    next(r for r in app.radio if r.label == "Input source").set_value("Supplied path example").run()
    app.checkbox(key="general_language").check().run()
    app.button(key="general_solve").click().run()
    assert not app.exception and not app.error
    saved = app.session_state.general_analysis
    assert saved["features"] is result and saved["concepts"] is None
    assert saved["concept_error"] == "Invalid concept JSON"
    assert released == [True]
    assert any("visual decisions below are retained" in value.value for value in app.warning)
