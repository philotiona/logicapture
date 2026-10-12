# Acceptance evidence

| Criterion | Status / evidence |
|---|---|
| Both modes in one interface | Implemented in app.py; both flows exercised by Streamlit AppTest and local browser |
| Akinator without retraining | Immutable AST predictions; answer filtering and undo tests |
| Hypothesis reduction | Count, trace chart, top candidates and finite-pool behavior count |
| Controlled page / two groups | Pixel-only demo, full-page uploads, separate uploads, snapshot widget |
| Objects and ≥5 relations | OpenCV hierarchy extraction; left/right, above/below, near, inside, size, shape/color, touching/overlap and collinearity |
| Ranked executable rules | Exact/noisy scoring, bounded composition, canonicalization, cached values and ambiguity counts |
| Explain matches/violations | Object selectors, counts, recursive evaluation, pair witnesses/violations and original-image overlays |
| Positive/counterexample drawings | Bounded rejection sampling; each result checked by its executable rule |
| Hidden examples | One panel per group withheld; app and experiment runner report results separately from support fit |
| Active versus random | Same finite pool, candidates and target programs; question caps and survivor traces logged |
| Neural versus symbolic | Fresh Keras encoder trained and evaluated alongside pixel, ground-truth and CV symbolic methods; historical artifact absence documented |
| Neural object perception | OpenCV individual regions → small TensorFlow shape/fill distributions → argmax hard attributes; retained OpenCV-only baseline |
| Probability-ready scenes | Validated optional distributions round-trip in JSON; hard reasoning uses argmax; future soft evaluator remains unimplemented |
| Real hand-drawn problems | **Broader independent data pending**; one supplied arrow-path screenshot regression now tested; template, manifest, protocol and evaluation CLI supplied |
| Important failures | Raw per-problem results, perception metrics, failure list and honest finite-domain limits |

This is a functioning implementation and synthetic pilot, not a completed human-drawing study. Camera hardware/browser permission and real photograph robustness have not been verified. The full course acceptance checklist remains open until real human drawings are supplied and tested.
