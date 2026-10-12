# Bongard reasoning from structure, with unknown evidence

The supplied arrow problem reveals a representation failure. A bent ink line is
not a malformed triangle. The arrows describe motion along a shared path; their
screen directions alone do not determine whether that motion agrees. A shape
classifier trained on circles and polygons cannot recover a distinction that
its output vocabulary does not represent.

This revision adds an evidence representation for paths and arrow marks. The
existing geometric scene representation, trained shape CNN, whole-image baseline,
and symbolic Akinator remain available. The new branch does not train another
CNN or silently relabel unusual paths as polygons.

## Representation and inference

```text
Original pixels
  → panel detection and crop validation
  → individual ink regions
  → evidence about both closed regions and stroke structure
       closed-region geometry / optional shape and fill CNN
       centerline, topology, arrow locations and tangent-relative direction
       unknown when the applicable evidence cannot be established
  → typed scene and measured evidence
  → executable predicates, counts, relations and Boolean programs
  → complete separating hypothesis, ambiguity, or insufficient evidence
  → explanations that identify the supporting paths and arrow marks
```

Perception operates on a single image without its Group A/B label. Search sees
the resulting scenes and labels. Human expected-arrow annotations are used only
for evaluation. A filename, remembered solution, a drawing's position in the
contact sheet, or a supplied rule sentence is not a perception feature.

The geometric confidence scores are evidence-quality heuristics. They are not
calibrated probabilities of correctness. Retained probability fields support a
future calibrated uncertainty model; their presence does not establish one.

`PathEvidence` retains an ordered normalized centerline, topology, width,
confidence, endpoints, junctions, graph segments and cycle counts, alongside
localized `ArrowMark` evidence. Direction is +1 or −1 relative to
that path's traversal, with 0 reserved for an undetermined direction. Reversing
the centerline reverses all known signs. The equality or opposition between two
arrow signs therefore remains unchanged. Rotation, reflection and translation
also preserve that relation, although they change the arrows' screen vectors.

The parser traces a thinned centerline and measures localized widening and taper
along it. Width variation alone is not an arrow: sharp bends can also widen a
normal cross-section. The detector rejects those bends and requires directional
silhouette evidence. Quarter-turn/reflection measurements reduce pixel-grid
bias; contradictory resolved interpretations remain unknown. This procedure
uses measured geometry rather than a fixed triangle label for an arrow's ink.

Open, closed and branched structures require different treatment. An ellipse can
be a closed directed path rather than an object whose only relevant attribute is
“circle.” A dense crossing or broken curve may not admit a reliable traversal.
Its arrow relation stays unknown; a plausible shape label does not fill this gap.
The measured curve is retained so the app can show what it actually interpreted.

The DSL supports path kind, topology, geometry, endpoint/junction/cycle counts,
arrow presence, arrow count and same/opposed flow. These predicates compose with the established quantifiers and
logical operators. For the supplied problem, a candidate concept is “a path has
arrows pointing in opposite directions along it.” An equivalent fit such as
“there is no path with arrows pointing the same way” should be exposed as an
alternative whenever the examples do not distinguish it. More examples, rather
than a higher fitting score, are needed to resolve that ambiguity.

## Unknown does not mean Group B

`Rule.assess(scene)` returns True, False or None. None means the concept cannot
be evaluated from the available evidence. Its negation is still None. Existing
`evaluate(scene)` remains a Boolean compatibility API for known symbolic scenes;
path validation and uploaded-image inference use `assess` to retain abstention.

An exact count over uncertain predicates uses lower and upper bounds. If there
are two certain matches and one unknown object, an exact count of two is unknown;
an exact count of four is false. Conjunction and disjunction can conclude only
when their known operands suffice. Search reports known coverage separately from
correctness. A rule that gets ten known panels right while ignoring two unknown
panels is not a twelve-panel explanation.

Abstention is necessary when arrow marks cannot be localized, the path traversal
is unresolved, an input is cropped incorrectly, or the separating concept is
outside the grammar. A softmax maximum from the old closed-set CNN is not an
out-of-distribution detector. Its retained distributions may inform later soft
reasoning, but cannot establish that a scribble belongs to its training vocabulary.

## Validation performed and how to reproduce it

`data/fixtures/arrow_paths` contains twelve unannotated cell crops from the user's
2026-10-11 screenshot. It includes the original ink and residual ruling, not the
app's mistaken triangle overlay. This is one problem with a known intended rule.
It is a regression fixture, not an independent estimate of Bongard performance.
The original full-resolution scan was not supplied.

Isolated outer ruling is removed by a geometric heuristic and recorded in the
scene warnings. A genuine large L-shaped drawing in the same position can be
ambiguous with that ruling. Original pixels and review overlays must remain
available; reliable background removal for arbitrary pages requires border/crop
context or a user correction. This limitation is separate from arrow reasoning.

The independent evaluator loads these pixels, extracts evidence without labels,
fits programs, and records:

1. Per-panel measured arrow counts and same/opposed relations, scored against
   annotations after extraction.
2. Six leave-one-pair-out fits. Each fit excludes one A and one B panel, then
   classifies only those two unseen panels with the induced program.
3. Fixed-program predictions after quarter-turn rotations, horizontal/vertical
   reflection, translation with unequal white padding and 75%/150% resizing of
   the actual pixels. The evaluator does not
   refit to transformed test images, which could conceal a shortcut.
4. Reordered support examples and swapped A/B labels. Perception is identical;
   the concept learner must respond to the new contrast.
5. Coverage, abstained count, accuracy among answered cases, and correct/total.
   All-abstaining cases have undefined conditional accuracy, not 100% accuracy.
6. Blank panels and identical observed evidence given contradictory labels. A
   ranking may contain partial candidates, but neither condition can receive a
   complete separating explanation.

Run:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.structural_regression
```

The JSON report is written to `data/regressions/structural_regression.json`. Transformed
images are dependent cases from the same twelve drawings. Good results on these
checks show that this particular regression and selected invariances work; they
do not establish robustness to new arrow styles or new Bongard concepts.

The supplied regression currently produces:

| Check | Result |
|---|---|
| Original arrow count and direction | 12/12 panels, two arrows per panel |
| Original open/closed topology | 12/12, including both closed loops |
| Leave-one-pair-out classification | 12/12 held-out predictions across six fits |
| Eight additional pixel transformation conditions | 96/96 predictions, full coverage |
| Reordered examples and swapped labels | 12/12 in each condition |
| Blank and contradictory input controls | No complete separating rule |

The best program is `Exists(arrow_relation="opposed")`. Fifteen programs fit the
original support examples perfectly, so the fit does not identify a unique
logical quantifier or count condition. This result supports the intended flow
distinction on these drawings while retaining the ambiguity about broader cases.

## Further concepts and measured limits

This representation makes extension possible without adding one closed-set CNN
class for every unusual drawing. The implemented graph retains endpoints,
junctions, segments and cycles; future operators can add enclosed-region holes,
containment, intersection, symmetry, concavity and repetitions. Those operators need independently annotated examples
and the same held-out and uncertainty tests. The current path branch does not
claim that every such property is implemented or robustly measured.

The next meaningful benchmark should contain independently authored problems,
not only different random seeds or transformed copies. It should separate:

| Measurement | Why it matters |
|---|---|
| Region/path detection precision and recall | Merged and missing ink cannot be repaired by rule search |
| Topology and arrow localization accuracy | A correct sentence can conceal incorrect local perception |
| Direction accuracy on independently varied arrow styles | The screenshot's style is not a universal arrow vocabulary |
| Held-out concept classification and coverage | Support fit alone rewards accidental separators |
| Exact or equivalent intended-program recovery | Multiple perfect fits can express different concepts |
| Abstention on blank, damaged and unsupported problems | Unknown evidence should not produce confident invented rules |
| Failures by concept family and image condition | Aggregate accuracy conceals unsupported families |

For the broader system, preserve four experimental conditions: whole-image CNN,
OpenCV with hard symbolic reasoning, neural perception with hard symbolic
reasoning, and eventual neural perception with soft symbolic reasoning. Add the
structural evidence branch as an explicit measured condition. Do not pool its
single-problem regression result into the earlier synthetic pilot or claim the
future soft evaluator has been implemented.

## Research basis

Bongard-LOGO explicitly separates free-form action strokes, basic shape concepts
and abstract attributes. Its authors emphasize that the same drawing can require
different interpretations according to the contrast. This supports treating
“triangle versus circle” as one available representation, rather than the entire
space of Bongard concepts. See the [original NeurIPS 2020 paper](https://papers.nips.cc/paper/2020/file/bf15e9bbff22c7719020f9df4badc20a-Paper.pdf).

A diagnostic study using ground-truth symbolic programs on Bongard-LOGO reports
large gains when structure is supplied. This is evidence that representation can
be a bottleneck; the paper explicitly treats symbolic input as a diagnostic upper
bound, not as a solved pixel-to-structure pipeline. See [Vaishnav and Tammet,
CoNLL 2026](https://aclanthology.org/2026.conll-main.19/).

Research evaluating multimodal language models also reports difficulty on
classical synthetic Bongard problems. Replacing the current pipeline with an
unvalidated model-generated explanation would not itself establish reliable
concept recovery. See [Małkiński et al., ICML 2025](https://proceedings.mlr.press/v267/malkinski25a.html).
