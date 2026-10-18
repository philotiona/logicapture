# Broader local Bongard inference

The general route does not enumerate complete rule sentences or force an unusual
curve into a circle/triangle/square class. It combines three complementary kinds
of evidence. It remains a research solver with measurable failures, not a claim
to solve every possible Bongard problem.

## Representations and inference

1. A frozen, revision-pinned SigLIP 2 vision encoder produces global image and
   dense patch features. No class names or query labels are required.
2. OpenCV measurements describe topology, counts, holes, nesting, boundaries,
   concavity, spectra, arrangement, morphology, symmetry, texture and orientation.
   The 544 numeric measurements have explicit validity indicators. They are not
   544 predefined Bongard rules. Their continuous combinations support concepts
   outside the executable DSL.
3. Within a problem, the classifier chooses feature views, support-only scaling,
   contrast weighting and prototype/ridge/kernel learners using nested support
   validation. Numeric magnitude is retained for quantitative concepts. Frozen
   parameters classify the unseen images. Scores are uncalibrated evidence,
   never displayed as correctness probabilities.

Unstable support fits, weak validation, contradictory supports, small margins
and visual novelty produce unresolved answers. These checks are empirical
diagnostics, not statistical correctness guarantees. Forced predictions and
answer coverage must be reported alongside accuracy on answered queries.

The existing executable DSL remains a separate specialist. A learned visual
decision boundary is not exported as an invented logical program. The original
arrow-flow and Akinator regressions continue to test their dedicated engines.

## Open-language concept search

Optional Qwen3 VL runs locally. The 2B profile supports CPU/GPU; the stronger 8B
profile uses NF4 quantization and CUDA, with an explicit memory budget. On this
RTX 5060, the 8B smoke test allocated approximately 6.4 GB of PyTorch tensors;
runtime/display memory is additional. Large concurrent models can exhaust VRAM.

The proposer sees only labelled supports. For drawing-like supports it can also
receive a compact table of measured contrasts selected from those supports.
Photo-like supports disable ink-geometry grounding. No primitive shape label,
dataset concept, filename, action annotation or query truth is supplied.

Each candidate is checked against one support image per prompt with the group
hidden. Failed candidates can be revised using support-only failure feedback.
The search has a visible finite round budget and deduplicates candidates. Only
candidates passing all support checks can classify unseen images. Conflicting
or unknown votes remain unresolved. Query images never enter proposal or repair.

These are group-blind checks on the same supports used to propose a concept,
not independent statistical validation. The same model can make correlated
errors while proposing and checking. Passing support checks does not establish
the intended rule; held-out queries are evaluated separately. Model-generated
text is parsed as bounded JSON and is never executed as code.

## Local setup and data

Install `requirements-general.txt`, use an appropriate official PyTorch wheel
for the GPU, and explicitly download model weights:

```powershell
python -m src.vision.pretrained --download siglip --download qwen
python -m src.vision.pretrained --download qwen8b  # optional stronger profile
python -m src.datasets.fetch_classic data/external/classic_bongard
```

Normal inference uses `local_files_only=True`, trusted built-in Transformers
classes and safetensors; no remote Python or hosted inference executes. Official
models are Apache-2.0. The exact repository revisions are in
`src/vision/pretrained.py`. Caches depend on canonical visible pixels, model
revision and preprocessing version; they contain no labels or solutions.

Transparent images are composited on white and EXIF orientation is applied
consistently. Corrupt feature caches are recomputed and writes are atomic.
Byte-identical visible support/query images are rejected by the general API.

## Evaluation

The benchmark runner accepts classic, HOI, LOGO and OpenWorld adapters. See
[DATASETS.md](DATASETS.md) for access, licensing and exact selection protocols.
HOI's roughly 50 GB image archive was not downloaded; annotation parsing alone
is not HOI image accuracy. Classic pages and OpenWorld photographs are evaluated
locally with hidden queries and documented availability exclusions.

```powershell
python -m src.evaluation.general_benchmark --partition test --jobs 4 --output outputs/classic_test_reproduction.json
python -m src.evaluation.general_benchmark --dataset openworld --root data/external/bongard_openworld_expanded --cohort data/external/bongard_openworld_expanded/cohort_test_expanded.json --partition test --jobs 2 --output outputs/openworld_test_reproduction.json
```

Every run freezes configuration, source hashes, source-image hashes, crop
geometry, support/query membership and provenance before prediction. Resume
rejects changed inputs or code. Query labels enter only post-prediction scoring.
Summaries include the global-prototype baseline, eligible-query denominators,
availability exclusions, strict duplicate exclusions and full-cohort answer
coverage. `structural` in these reports means numeric features plus episodic
learning, not the separate executable DSL engine.

Frozen availability cohorts run in full. For a smaller language pilot, freeze an
explicit subcohort before inference; `--limit` applies to non-cohort loaders.

Measured results, including failed language hypotheses and the weaker
natural-image results, are published in [the external evaluation report](../data/benchmarks/REPORT.md).

Public problems may have occurred in pretrained model data. There is no claim
that withholding a cell proves absence of pretraining contamination, nor that
held-out image accuracy measures recovery of the intended human explanation.

Primary background: [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224),
[Qwen3 VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), and
[Support-Set Context Matters](https://arxiv.org/abs/2309.03468). This implementation
uses its own episodic learner and does not claim to reproduce those papers.
