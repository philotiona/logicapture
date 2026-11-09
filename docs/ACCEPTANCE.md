# Current product acceptance

See [the product brief](PROJECT_BRIEF.md) for the current scope. Historical game and benchmark results do not define the application navigation.

| Criterion | Verification |
|---|---|
| Bongard is the opening screen; no game or experiment route | `tests/test_app.py` checks navigation and migration of retired sessions |
| No algorithm, representation, CNN, language-model or holdout selectors | `tests/test_app.py` and `tests/test_general_interface.py` inspect the available controls |
| One action produces an explanation and two checked new drawings | The supplied arrow and generated boundary cases in `tests/test_general_interface.py` |
| All given examples remain available to inference | Three-per-group upload and twelve-panel page tests |
| Unsupported problems remain unsolved | Explicit no-rule test; no feature-classification substitution |
| New queries use frozen rules and reference copies are identified | New-image test without refitting |
| Changed inputs hide stale results | Upload and example changes in the interface tests |
| Full-page input uses the same rule route | Dotted-page upload in `tests/test_page_live_workflow.py` |
| Image sorting remains practical and independent | Task save/reopen, review, export and invalidation tests |
| Coursework remains separate and usable | РГР interface, photo, training and report tests |

Scene semantics, rule search, path extraction, constructive generation, optional neural contracts and historical evaluation protocols retain their backend tests. They do not add product controls.
