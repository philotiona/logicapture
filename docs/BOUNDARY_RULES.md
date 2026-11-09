# Boundary geometry rule recovery

The main Bongard solver now measures simplified-boundary convexity and rotation-independent elongation. These executable observations extend the existing pixel-to-rule route for uploaded pages, separate images and camera analysis.

Try **Bongard → Try an example**, then **Convex and concave shapes** or **Long and compact shapes**, and press **Find the rule**. The demonstrations vary shape, rotation and fill. Only rendered pixels and A/B membership enter inference; the generator's concept name is not passed to the solver. The product now uses all six examples per group; independently uploaded new drawings test the frozen rule afterward. New matching drawings and counterexamples are rendered and re-read; every frozen program must agree with their labels. The saved audit below retains its original held-out protocol.

## Measurements

The original raw solidity features remain available. Two additional features measure minimum and median area/convex-hull-area ratios after simplifying each boundary at a fixed 2% of its perimeter. This reduces raster stair-step effects on small convex figures. Indentations below that resolution can still disappear, so simplified-boundary convexity is a measured approximation.

Three further features measure minimum, median and maximum long-side/short-side ratios of each component's minimum-area rotated rectangle. A diagonal thin figure remains elongated even when its horizontal and vertical extents are similar. Degenerate or boundary-clipped evidence remains unavailable.

Thresholds are refitted while withholding one support from each group. Qualifying alternatives are retained; a query requires their agreement. The support checks also select features, so their agreement is not independent test accuracy. Exports record the observation version and boundary tolerance.

## Whole-collection comparison

Both runs use the same local Foundalis manifest and last-cell-per-group query reservation. All 408 static pages are split. Twelve animated images belonging to one additional problem remain unsupported.

| Result | Before | After |
|---|---:|---:|
| Problems with an executable candidate | 33 | 36 |
| Answered query decisions | 66 | 72 |
| Correct answered decisions | 57 | 62 |
| Wrong answered decisions | 9 | 10 |
| Problems with both withheld decisions correct | 27 | 29 |
| Static problems without a candidate | 375 | 372 |

All previously answered decisions are unchanged, including their errors. Added candidates:

| Problem | Recovered measurement | Correct withheld decisions |
|---|---|---:|
| 4 | Simplified-boundary convexity | 2 / 2 |
| 12 | Rotation-independent elongation | 2 / 2 |
| 152 | Rotation-independent elongation | 1 / 2 |

Answer coverage increases; precision among answered queries changes from 86.4% to 86.1%. The extra error is retained. Problem 152 illustrates that a stable support separator can reflect an accidental property. Correct query classification does not establish recovery of the human-intended concept.

The measurements were developed after inspecting collection examples. This is a development coverage audit, not a newly unseen benchmark or an independent representation-selection evaluation. Most problems remain unresolved.

Raw evidence: [baseline](../data/benchmarks/foundalis_boundary_baseline.json), [updated audit](../data/benchmarks/foundalis_boundary_extension.json), [comparison and all added decisions](../data/regressions/boundary_rules.json). Source/image hashes and failures are retained.

Reproduce the current audit:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.foundalis_audit --workers 4 --output outputs/boundary_audit.json
```

Tests cover small filled/outline polygons, rotation, missing/clipped evidence, inverted labels, pixel-only inference, checked generation, default-screen interaction and the existing arrow/path flow.
