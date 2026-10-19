# Complete pages, local camera, and measured rule coverage

The default Bongard input is now **Page image**. Upload a complete image with
positive examples on the left and negative examples on the right. The app finds
observed panel frames, determines equal group counts, orders the crops by row,
and shows the extracted groups before inference. Manual cropping is optional.
The supplied dotted page is also available as a reproducible example.

Each detected quadrilateral can be rectified separately. A visible outer paper
boundary is optional. One incomplete frame can be repaired only when eleven
other similarly sized frames establish a consistent two-column, three-row
layout on each side. Inferred boundaries are reported for review. A blank page
does not produce an invented grid. Borderless, obscured, or unconventional
layouts can still require correction; successful frame detection is not proof
that every mark was retained correctly.

## Live camera

Install the optional transport dependency and start the local app:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-camera.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Choose **Live camera**, start video, and allow camera access in the browser.
Fill most of the frame with the complete problem. Audio is disabled. Browser
camera access needs localhost or HTTPS. The app configures no hosted model,
STUN, or TURN service; this setup targets a browser and server on the same
computer. Camera hardware and OS/browser permissions remain necessary.

The processor samples frames, detects panels, and waits for a stable view.
Executable drawing inference runs in one background worker, leaving video
processing responsive. The returned video contains detected boundaries, the
candidate rule, and a new positive and negative drawing when both can be
constructed and re-read successfully. Changing or losing the page invalidates
old results, including late worker completions. Sparse drawings are compared
relative to their ink rather than their mostly white background.
An unsuccessful stable reading can be retried on fresh frames twice, with a
two-second interval; compressed startup frames do not permanently freeze a bad
reading, and unsuccessful inference does not run in an unbounded loop.

The local WebRTC round-trip test sends camera-like frames through real codecs
and returns the annotated video with a checked generated pair. It passed four
fresh connections after the startup retry fix. A physical camera could not be
verified in the in-app browser on this machine: it reported **Requested device
not found**. This is separate from the transport and image-processing tests.

**Use current camera page for full analysis** freezes the current crops for
the broader solver. A single-frame capture is available as a fallback. If no
executable rule or checked generated pair is available, the app says so.
It does not label a copied support image as a newly generated example.

## Executable measurements and generated examples

The drawing route now complements the existing compositional region/path DSL
with named numeric rules: component and hole counts, ink density, size and
extent, convexity, circularity, symmetry, and relative dot spacing. Dot spacing
uses dot-width units and preserves unknown counts when edge marks are clipped.
This measures relationships without assigning a primitive shape class to every
mark. It remains a bounded set of observations, not arbitrary visual cognition.

Measurement thresholds are refitted with one support omitted from each group.
Feature selection uses those support checks; their displayed agreement is
therefore **not independent test accuracy**. All equally simple qualifying
measurement rules are retained. A query that separates those alternatives, or
needs unavailable evidence, is unresolved. Frozen queries are evaluated only
after fitting. A hypothesis may still match the supports for the wrong reason.

The supplied dotted-page regression uses five supports per group, classifies
both withheld panels correctly, and constructs a new close-dot positive and
spread-dot negative. Tests also reverse group labels. Every generated drawing
is rendered and re-read from pixels, and every frozen candidate must agree with
its assigned label. This verifies the recovered program, not the intended
human interpretation of the original problem.

## Foundalis coverage

The complete frozen [Foundalis index](https://www.foundalis.com/res/bps/bpidx.htm)
contains **409 numbered entries**. All were retrieved locally without fetching
the solutions page: 408 static complete-page images, plus twelve animated
panels belonging to problem 300. All 420 image files are accounted for. Media
remain ignored by Git; no redistribution license is inferred.

The final executable-rule audit is in
[the raw report](../data/benchmarks/foundalis_rules_audit.json). It withholds the
last panel per group before rule fitting. Solver inputs contain only pixels and
group membership; source names, IDs, and explanations are not supplied.

| Result | Count |
|---|---:|
| Static pages automatically split into six positive and six negative crops | 408 / 408 |
| Static problems with a support-checked executable candidate | 33 / 408 |
| Withheld query decisions answered by those candidates | 66 |
| Correct answered query decisions | 57 / 66 |
| Problems with both withheld query decisions correct | 27 / 408 |
| Static problems without an accepted executable candidate | 375 / 408 |
| Animated problem requiring temporal reasoning | 1 |

**The requirement to recover every Foundalis rule has not been achieved.**
Nine answered query decisions are wrong, and most problems lack an executable
candidate. Two problems have support/query pixel duplicates and are recorded
without claiming independent validation. Unknown rules, unusual perception,
meta-concepts and temporal concepts remain major limitations. No intended-rule
accuracy is claimed. Neither crop coverage nor successful held-out
classification proves recovery of an intended rule.

This is a development coverage audit of the public collection, not a newly
unseen benchmark or a measurement of the optional vision-language model. The
older feature-classifier benchmarks are separate experiments. Source hashes,
image hashes, per-problem failures, candidate programs and query results are
retained so this boundary can be inspected.

```powershell
python -m src.datasets.fetch_foundalis data/external/foundalis --workers 4
python -m src.evaluation.foundalis_audit --workers 4 --output outputs/foundalis_audit.json
```
