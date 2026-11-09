from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from src.interface import general_bongard as interface
from src.reasoning.bongard import BongardResult
from src.vision import pretrained

ROOT = Path(__file__).resolve().parents[1]


class Upload(BytesIO):
    def __init__(self, path):
        super().__init__(path.read_bytes())
        self.name = path.name


def test_one_find_rule_action_uses_all_examples_and_generates_pixel_checked_answers(monkeypatch):
    def no_weights(*args, **kwargs):
        raise AssertionError("The product's Bongard workflow must not require model downloads or statistical classification")
    monkeypatch.setattr(pretrained, "SiglipEncoder", no_weights)
    monkeypatch.setattr(pretrained, "QwenLocalBackend", no_weights)
    app = AppTest.from_file(ROOT/"app.py", default_timeout=45).run()
    app.radio(key="bongard_source").set_value("Try an example").run()
    app.button(key="bongard_solve").click().run()
    assert not app.exception and not app.error
    result = app.session_state.bongard_analysis["result"]
    assert result.model is not None and "opposite ways" in result.diagnostics["logical_explanation"]
    assert result.diagnostics["structural_attempt"]["support_total"] == 12
    assert result.diagnostics["all_given_examples_used"]
    assert len(result.examples) == 2 and all(example.verification["pixel_reread"] for example in result.examples)
    assert not app.checkbox and not app.slider
    assert not any(radio.label in ("Reasoning approach", "Perception pipeline", "Representation") for radio in app.radio)
    assert not any(button.key == "bongard_solve" for button in app.button)
    app.selectbox(key="bongard_example").set_value("Long and compact shapes").run()
    assert app.button(key="bongard_solve") and not any('<div class="rule-output">' in markdown.value for markdown in app.markdown)


@pytest.mark.parametrize("name", ["Convex and concave shapes", "Long and compact shapes"])
def test_geometry_examples_use_the_same_automatic_route(name):
    app = AppTest.from_file(ROOT/"app.py", default_timeout=45).run()
    app.radio(key="bongard_source").set_value("Try an example").run()
    app.selectbox(key="bongard_example").set_value(name).run()
    app.button(key="bongard_solve").click().run()
    assert not app.exception and not app.error
    result = app.session_state.bongard_analysis["result"]
    assert result.diagnostics["selected_pipeline"] == "executable_measurement_rules"
    assert result.diagnostics["structural_attempt"]["support_total"] == 12
    assert len(result.examples) == 2


def test_three_uploaded_examples_are_not_silently_withheld_and_changed_pixels_hide_old_result(monkeypatch):
    files = {f"bongard_{letter.lower()}": [Upload(ROOT/f"data/fixtures/arrow_paths/{letter}{i:02d}.png") for i in range(1, 4)]
             for letter in "AB"}
    monkeypatch.setattr(interface.st, "file_uploader", lambda *args, key=None, **kwargs: files.get(key, []))
    monkeypatch.setattr(DeltaGenerator, "file_uploader", lambda self, *args, key=None, **kwargs: files.get(key, []))
    app = AppTest.from_file(ROOT/"app.py", default_timeout=45).run()
    app.radio(key="bongard_source").set_value("Separate images").run()
    app.checkbox(key="bongard_framed").check().run()
    app.button(key="bongard_solve").click().run()
    assert not app.exception and not app.error
    assert app.session_state.bongard_analysis["result"].diagnostics["structural_attempt"]["support_total"] == 6
    app.checkbox(key="bongard_framed").uncheck().run()
    assert app.button(key="bongard_solve")
    app.checkbox(key="bongard_framed").check().run()
    files["bongard_a"] = [Upload(ROOT/f"data/fixtures/arrow_paths/A{i:02d}.png") for i in range(2, 5)]
    app.run()
    assert not any('<div class="rule-output">' in markdown.value for markdown in app.markdown)
    assert app.button(key="bongard_solve")


def test_new_queries_use_the_frozen_rule_without_refitting_and_mark_reference_copies(monkeypatch):
    files = {"bongard_queries": []}
    monkeypatch.setattr(interface.st, "file_uploader", lambda *args, key=None, **kwargs: files.get(key, []))
    calls, infer = [], interface.infer_problem
    def capture(*args, **kwargs):
        calls.append(True)
        return infer(*args, **kwargs)
    monkeypatch.setattr(interface, "infer_problem", capture)
    app = AppTest.from_file(ROOT/"app.py", default_timeout=45).run()
    app.radio(key="bongard_source").set_value("Try an example").run()
    app.button(key="bongard_solve").click().run()
    result = app.session_state.bongard_analysis["result"]
    for index, example in enumerate(result.examples):
        stream = BytesIO()
        Image.fromarray(example.image).save(stream, format="PNG")
        stream.name = f"unseen-{index}.png"
        files["bongard_queries"].append(stream)
    files["bongard_queries"].append(Upload(ROOT/"data/fixtures/arrow_paths/A01.png"))
    app.run()
    assert not app.exception and not app.error and len(calls) == 1
    table = app.dataframe[-1].value
    assert table["Result"].tolist() == ["Group A", "Group B", "Given example"]
    app.checkbox(key="bongard_query_framed").check().run()
    assert not app.exception and len(calls) == 1
    assert {key[1] for key in app.session_state.bongard_analysis["query_cache"]} == {False, True}


def test_unsolved_problem_never_substitutes_a_feature_score_for_a_recovered_rule(monkeypatch):
    monkeypatch.setattr(interface, "infer_problem", lambda *args, **kwargs:
                        BongardResult(None, {"structural_attempt": {"reason": "no_consistent_structural_rule"}}, ()))
    app = AppTest.from_file(ROOT/"app.py", default_timeout=30).run()
    app.radio(key="bongard_source").set_value("Try an example").run()
    app.button(key="bongard_solve").click().run()
    assert not app.exception and not app.error
    assert any("No verified rule" in message.value for message in app.warning)
    assert not any('<div class="rule-output">' in markdown.value for markdown in app.markdown)


def _capture_reports(monkeypatch):
    reports, build = [], interface._report_payload
    def capture(*args, **kwargs):
        payload = build(*args, **kwargs)
        reports.append(json.loads(json.dumps(payload, default=str)))
        return payload
    monkeypatch.setattr(interface, "_report_payload", capture)
    return reports


def _has_report_button(app):
    return any(getattr(element, "label", None) == "Download rule report"
               or getattr(getattr(element, "proto", None), "label", None) == "Download rule report"
               for element in app.get("download_button"))


def test_unsolved_problem_still_offers_its_rule_report(monkeypatch):
    attempt = {"reason": "no_consistent_structural_rule", "accepted": False}
    monkeypatch.setattr(interface, "infer_problem", lambda *args, **kwargs:
                        BongardResult(None, {"structural_attempt": attempt}, ()))
    reports = _capture_reports(monkeypatch)
    app = AppTest.from_file(ROOT/"app.py", default_timeout=30).run()
    app.radio(key="bongard_source").set_value("Try an example").run()
    app.button(key="bongard_solve").click().run()
    assert not app.exception and not app.error
    assert _has_report_button(app)
    report = reports[-1]
    assert report["rule_found"] is False
    assert report["diagnostics"]["structural_attempt"] == attempt
    assert report["predictions"] == [] and len(report["support_pixel_hashes"]) == 12


def test_report_records_every_query_shown_in_the_table(monkeypatch):
    from src.active.path_generation import path_object
    from src.scene import Scene
    from src.scene.renderer import draw_scene
    files = {"bongard_queries": []}
    monkeypatch.setattr(interface.st, "file_uploader", lambda *args, key=None, **kwargs: files.get(key, []))
    reports = _capture_reports(monkeypatch)
    app = AppTest.from_file(ROOT/"app.py", default_timeout=45).run()
    app.radio(key="bongard_source").set_value("Try an example").run()
    app.button(key="bongard_solve").click().run()
    result = app.session_state.bongard_analysis["result"]
    def png(pixels, name):
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        stream.name = name
        return stream
    broken = BytesIO(b"not an image")
    broken.name = "broken.png"
    files["bongard_queries"] += [png(result.examples[0].image, "unseen.png"),
                                 Upload(ROOT/"data/fixtures/arrow_paths/A01.png"), broken,
                                 png(np.array(draw_scene(Scene([path_object("wave", ())]))), "plain-curve.png")]
    app.run()
    assert not app.exception and not app.error and _has_report_button(app)
    assert app.dataframe[-1].value["Result"].tolist() == ["Group A", "Given example", "Unreadable", "Cannot decide"]
    report = reports[-1]
    assert report["rule_found"] is True and report["diagnostics"]["structural_attempt"]["accepted"]
    records = report["predictions"]
    assert [r["filename"] for r in records] == ["unseen.png", "A01.png", "broken.png", "plain-curve.png"]
    assert [r["status"] for r in records] == ["predicted", "given_example", "unreadable", "abstained"]
    assert all(r["reason"] for r in records)
    assert records[1]["label"] is records[2]["label"] is records[3]["label"] is None
    predicted = records[0]
    assert predicted["label"] is True
    for field in ("image_id", "forced_label", "score", "probability", "abstention_reasons",
                  "nearest_support_ids", "diagnostics"):
        assert field in predicted
    assert records[3]["abstention_reasons"] and "diagnostics" in records[3]


def test_page_layout_corrections_reset_when_a_different_page_is_uploaded(monkeypatch):
    files = {"page": BytesIO((ROOT/"data/fixtures/dotted_page.png").read_bytes())}
    monkeypatch.setattr(interface.st, "file_uploader", lambda *args, key=None, **kwargs:
                        files["page"] if key == "bongard_page" else [])
    app = AppTest.from_file(ROOT/"app.py", default_timeout=30).run()
    app.slider(key="bongard_divider").set_value(.45).run()
    assert app.slider(key="bongard_divider").value == .45
    with Image.open(files["page"]) as original:
        pixels = np.pad(np.array(original.convert("RGB")), ((8, 8), (8, 8), (0, 0)), constant_values=255)
    replacement = BytesIO()
    Image.fromarray(pixels).save(replacement, format="PNG")
    files["page"] = replacement
    app.run()
    assert not app.exception and not app.error
    assert app.slider(key="bongard_divider").value == .5
    assert app.selectbox(key="bongard_count").value == "Detect automatically"
