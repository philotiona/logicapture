from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]/"app.py"


def test_app_opens_bongard_without_game_benchmarks_or_algorithm_controls():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception and not app.error
    assert app.session_state.mode == "Bongard"
    assert {button.label for button in app.button} == {"Find a rule", "Local Lab", "Sort images", "РГР coursework"}
    assert [radio.label for radio in app.radio] == ["Problem input"]
    assert not app.checkbox and not app.slider and not app.selectbox
    assert any("Upload your problem" in info.value for info in app.info)


def test_retired_workspaces_migrate_to_bongard():
    for mode in ("Akinator", "Experiments", "Print Inspector"):
        app = AppTest.from_file(APP, default_timeout=30)
        app.session_state.mode = mode
        app.session_state.akinator = "retired session"
        app.session_state.active_seed = 42
        app.run()
        assert not app.exception
        assert app.session_state.mode == "Bongard"
        assert "akinator" not in app.session_state and "active_seed" not in app.session_state


def test_sorting_and_coursework_are_separate_from_rule_discovery():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key="nav_image_tasks").click().run()
    assert not app.exception
    assert app.session_state.mode == "Sort images"
    assert any(heading.value == "Sort images by your examples" for heading in app.subheader)
    app.button(key="nav_bongard").click().run()
    assert not app.exception and app.session_state.mode == "Bongard"


def test_local_lab_opens_from_top_bar_and_generates_without_a_model_service():
    app = AppTest.from_file(APP, default_timeout=45).run()
    app.button(key="nav_lab").click().run()
    assert not app.exception and app.session_state.mode == "Local Lab"
    assert app.radio(key="lab_source").value == "Try an example"
    app.button(key="lab_generate").click().run()
    assert not app.exception and not app.error
    assert any("fits the learned Group A contrast" in item.value for item in app.success)
    app.radio(key="lab_source").set_value("Upload a page").run()
    assert not app.success
