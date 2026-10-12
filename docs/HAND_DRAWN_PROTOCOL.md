# Real-drawing evaluation protocol

One real arrow-path problem has been tested from the user's supplied screenshot. Its twelve panel crops and dependent transform/holdout checks are a regression fixture, not a dataset of independently drawn problems. The remaining acceptance criterion still requires several independent human-authored pages with artist and capture metadata. See `BONGARD_REDESIGN.md` for the structural pipeline and its validation.

1. Print `data/bongard_template.png`. Make 10–20 problems from supported concepts; include several people and at least two lighting conditions. Draw six independent examples in each group.
2. Use closed circles, triangles and quadrilaterals with black marker; record fill, approximate centers, containment and intended rule. Avoid labels inside cells, broken strokes and occlusion for the first set.
3. Photograph or scan the page. Keep all corners/cell borders visible. Preserve the original photograph and an independently annotated ground-truth JSON scene for each panel.
4. Record relative paths and the typed target program in a manifest following `data/hand_drawn/manifest.example.json`. Obtain consent for retaining drawings; do not include people or private information in the images.
5. Use the app's **Complete page** or **Separate groups** input with holdout enabled. Each group's last panel is excluded from fitting. Export the analysis JSON.
6. Independently compare detected objects/attributes/relations with annotations. Record count precision/recall, shape/fill accuracy, held-out classification, exact program recovery and alternative fitting rules separately.
7. Repeat on the second lighting photograph, with adaptive thresholding as an explicitly recorded condition. If cropping or manual correction is needed, record it instead of treating it as autonomous perception success.
8. Preserve failures: merged ink, broken contours, crop loss, incorrect shape/fill, relation tolerance sensitivity, no consistent program, generation timeouts and ambiguous intended concepts.
9. Repeat each problem with both OpenCV-only and neural attribute perception. Keep the images and held-out split identical; export the hard attributes and distributions. A future soft evaluator is a separate condition and must not be claimed before it is implemented.

`python -m src.evaluation.hand_drawn MANIFEST --output outputs/hand_drawn` automates extraction, withholding panels, rule search and JSON/CSV reports for individually cropped group images. Complete pages can be segmented through the app first. The manifest accepts a serialized DSL AST for the intended rule and optional symbolic annotations. Data collection itself must come from people; synthetic deformation is not a replacement.
