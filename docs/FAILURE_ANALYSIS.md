# Failure analysis and interpretation

The user's arrow-path screenshot exposed a closed-set representation failure: a bent open line was forced into a triangle label, and shape/fill features could not express same/opposed travel along a curve. The default now retains paths, actual closed boundaries, graph topology and localized arrow directions. See `BONGARD_REDESIGN.md` and `data/regressions/structural_regression.json`. The historical pilot below remains a measurement of the earlier geometric pipeline.

The included pilot deliberately reports the limits of this bounded concept language and its perception front end. Use `data/pilot/results.json` for exact programs, panels, seeds, conditions and counts; `REPORT.md` for measured aggregate results. No real human drawings were evaluated.

## Ambiguous support

A program can fit every support panel and still disagree with the intended concept. Examples in the pilot include “exactly one circle” versus “contains a circle,” a relative-size relation instead of a conjunction of circle presence and object count, and an above relation instead of left-of. These alternatives often tie or beat the intended rule under a simplicity prior. Exact program recovery, agreement on 100 independently generated probes, and withheld-panel accuracy are recorded separately. A high held-out classification score is not proof that the intended program was recovered.

Counts are balanced across labels for basic existence families. Conjunction negatives violate each clause separately. Remaining accidental correlations and ambiguity are retained and reported rather than silently calling every best explanation the secret rule. The interface displays perfect-hypothesis counts and alternatives and can draw a disagreement scene.

## Perception

OpenCV can merge overlapping/touching ink into one component, lose broken outlines, discard a boundary-clipped object, or classify a deformed shape as a polygon. Contour hierarchy avoids double-counting outline interiors and keeps nested components, but it does not solve arbitrary topology. Shape/fill and relation metrics compare detections with explicit ground-truth objects; per-object rotation retains centers and updates the geometry before perception scoring.

The neural path shares segmentation with the deterministic baseline. It therefore cannot recover an object OpenCV failed to segment. Its small CNN was trained on isolated synthetic objects, including regular pentagons. Confidence distributions are uncalibrated and can be wrong on nested/thin/irregular real contours. Preserve these distributions and inspect the evidence; an argmax does not establish certainty. The baseline comparison can legitimately show deterministic perception outperforming neural perception in some conditions.

## Active learning

Stopping means the surviving rules agree on the finite query pool, not that exactly one logically distinct rule exists. Some different syntax trees or rare predicates remain inseparable. The interface permits adding fresh scenes. Random-query runs stop at 40 questions; their reported means are censored by that cap. Compare question counts together with completion/cap rates and traces. The updated active protocol may group up to four independent drawings after repeated negative answers; its drawing counts must be reported alongside questions. The included historical pilot predates this protocol.

## Generation and search

Rejection sampling can time out on impossible or rare concepts. This is an explicit error and never produces an unverified drawing. Counted attributes and total-count/universal-property rules now have constructive generators. Random scenes use 0–6 objects; deliberate count tests can include up to eight to separate matching counts from irrelevant distractors. Search uses a bounded grammar and a beam of 40 distinct prediction patterns for passive expansion. A perfectly representable AST is not necessarily enumerated by this restricted search. Depth alone is not a completeness guarantee.

## Neural baseline

The fresh whole-image CNN has a limited 180-episode pilot training budget. Low accuracy in this run must be reported; it is not evidence that all neural baselines are intrinsically weaker. The original trained model and its generator are absent. Larger training budgets, learning-curve comparisons, independent seeds and more held-out panels are needed before drawing broader conclusions.

## Remaining study

Collect actual hand-drawn pages, run identical splits through both perception paths, and include independent object/relation annotations. Soft/probabilistic symbolic evaluation remains planned. Its semantics and calibration must be tested before it becomes the fourth measured comparison system.
