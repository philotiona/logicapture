from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
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


def test_default_screen_recovers_arrow_rule_and_answers_hidden_queries(monkeypatch):
    def no_encode(*args, **kwargs):
        raise AssertionError("Default arrow reasoning must use the executable path solver")
    monkeypatch.setattr(interface, "semantic_encoder", lambda: SimpleNamespace(encode=no_encode))
    monkeypatch.setattr(pretrained, "model_directory", lambda *args: Path("."))
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=45).run()
    app.button(key="nav_bongard").click().run()
    # Do not switch to the specialist screen: this was the missing regression.
    assert next(r for r in app.radio if r.label == "Reasoning approach").value == "General visual inference"
    next(r for r in app.radio if r.label == "Input source").set_value("Supplied path example").run()
    assert app.checkbox(key="general_structural_first").value is True
    app.button(key="general_solve").click().run()
    assert not app.exception and not app.error
    result = app.session_state.general_analysis["features"]
    assert result.diagnostics["selected_pipeline"] == "executable_structural_rules"
    assert "opposite ways" in result.diagnostics["logical_explanation"]
    table = app.dataframe[0].value
    assert table["Rule decision"].tolist() == table["Held-out truth"].tolist()
    assert set(table["Measured path flow"]) == {"opposed", "same"}
    assert any("2/2 correct, 0 unresolved" in caption.value for caption in app.caption)


def test_three_image_upload_preserves_structural_support_and_resets_large_page_default(monkeypatch):
    from streamlit.delta_generator import DeltaGenerator

    folder = Path(__file__).resolve().parents[1]/"data/fixtures/arrow_paths"
    uploads = {f"general_{group.lower()}": [np.array(Image.open(folder/f"{group}{i:02d}.png").convert("RGB"))
                                           for i in range(1, 7)] for group in "AB"}
    monkeypatch.setattr(interface.st, "file_uploader", lambda *args, key=None, **kwargs: uploads.get(key, []))
    monkeypatch.setattr(DeltaGenerator, "file_uploader", lambda self, *args, key=None, **kwargs: uploads.get(key, []))
    monkeypatch.setattr(interface, "_load_upload", lambda image: image)
    monkeypatch.setattr(pretrained, "model_directory", lambda *args: None)
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=45).run()
    app.button(key="nav_bongard").click().run()
    next(r for r in app.radio if r.label == "Input source").set_value("Separate images").run()
    assert app.checkbox(key="general_holdout").value is True
    # Replacing the six-image upload with three images must reset its default.
    for key in uploads:
        uploads[key] = uploads[key][:3]
    app.run()
    assert app.checkbox(key="general_holdout").value is False
    app.checkbox(key="general_generate").uncheck().run()
    app.button(key="general_solve").click().run()
    assert not app.exception and not app.error
    result = app.session_state.general_analysis["features"]
    assert len(result.support_ids) == 6 and result.query_ids == ()
    assert result.diagnostics["selected_pipeline"] == "executable_structural_rules"
    assert result.diagnostics["structural_attempt"]["support_total"] == 6
    # A deliberate two-support test remains available and survives reruns.
    app.checkbox(key="general_holdout").check().run()
    assert app.checkbox(key="general_holdout").value is True
    assert any("leaves two supports" in caption.value for caption in app.caption)
    app.run()
    assert app.checkbox(key="general_holdout").value is True
    # Feature-only mode can reserve a test while retaining its two supports.
    app.checkbox(key="general_structural_first").uncheck().run()
    assert app.checkbox(key="general_holdout").value is True
    app.checkbox(key="general_structural_first").check().run()
    assert app.checkbox(key="general_holdout").value is False


def test_generated_geometry_uses_pixel_rules_and_checks_unseen_drawings(monkeypatch):
    monkeypatch.setattr(pretrained, "model_directory", lambda *args: None)
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=45).run()
    app.button(key="nav_bongard").click().run()
    next(r for r in app.radio if r.label == "Input source").set_value("Generated geometry").run()
    for name in ("Convex / concave boundaries", "Elongated / compact shapes"):
        app.selectbox(key="general_geometry").set_value(name).run()
        app.button(key="general_solve").click().run()
        assert not app.exception and not app.error
        saved = app.session_state.general_analysis
        result = saved["features"]
        assert result.diagnostics["selected_pipeline"] == "executable_measurement_rules"
        assert result.diagnostics["structural_attempt"]["support_total"] == 10
        assert result.diagnostics["structural_attempt"]["query_images_used_in_selection"] is False
        assert all(p.label is not None for p in result.predictions)
        assert any("2/2 correct, 0 unresolved" in caption.value for caption in app.caption)
        assert len(saved["generated"]) == 2
        assert all(example.verification["pixel_reread"] for example in saved["generated"])
