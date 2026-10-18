# Independent Bongard datasets and leakage boundaries

The project now has strict local-root adapters for classic Bongard pages,
Bongard-HOI, Bongard-LOGO and Bongard-OpenWorld. These adapters supply pixels to
the broader solvers without disclosing annotated concepts or query truth.
External data lives under ignored `data/external/`; it is not bundled into Git.
No upstream training scripts or downloaded Python loaders are executed.

## Available local evaluation data

| Dataset condition | Development | Test | Actual local data |
|---|---|---|---|
| Classic original problems | p0001–p0020;19 eligible after duplicate exclusion | p0021–p0100;80 eligible |100 original pages and1,200 cell crops |
| OpenWorld, expanded availability-filtered3-shot |15 eligible from first20 official validation episodes |37 eligible from first50 official test episodes |785 verified image files;195 URL failures logged |
| HOI | One real annotation episode for parser validation | No image-based measurement claimed | Small official annotation sample; photographs not downloaded |
| LOGO | Adapter ready for local official archive | No full-dataset measurement claimed | No full LOGO archive installed |

These are data-availability counts, not model accuracy. Model evaluation reports
must identify the encoder, representation, fitting policy and exclusions. The
classic development/test partition was fixed before test-set inference. Public
classic problems may have appeared in pretrained-model data; holding out cells
does not prove the pretrained model has never encountered a problem.

## Solver input contract

`DatasetEpisode` retains dataset and split identifiers, source annotation SHA256,
paths, fixed support/query membership and an optional hidden audit concept.
`episode.prepare(seed=0)` produces two separate objects:

```python
prepared.input       # SolverInput(positive, negative, queries): RGB uint8 arrays
prepared.query_labels  # evaluation truth only; never given to the solver
prepared.query_order   # evaluation bookkeeping only
```

Only `prepared.input` crosses the solver boundary. It contains no concept,
caption, action label, object label, filename, task name, query label, subject or
object box, or generation program. This matters especially for LOGO task names
and OpenWorld filenames, which contain readable concept or URL information.
Positive support means Boolean True in this project. The official HOI loader's
numeric class encoding is not reused.

Query order is shuffled reproducibly per episode and seed, so the first query
is not always positive. Supports and queries are never drawn from the same
source-image path, even if their crop boxes differ. By default preparation also
rejects byte-identical files at different paths. That conservative check catches
copied images; it does not establish absence of near-duplicates or pretraining
contamination. `audit_split_overlap` reports path reuse between split collections
separately instead of silently changing a published split.

Paths must remain within the configured root, including resolved symlinks. URI,
drive, UNC, traversal, reserved Windows names and ambiguous trailing-dot paths
are rejected. Files are re-resolved when read. Invalid crop geometry, malformed
JSON, duplicate JSON keys, missing images and repeated references cause explicit
errors. Metadata-only loading uses `require_files=False` and cannot be reported
as image evaluation.

## Classic original problems

The [Bongard in Wonderland repository](https://github.com/ml-research/bongard-in-wonderland)
links the authors' high-resolution [OSF data collection](https://osf.io/95dks/).
The public folder contains100 separately downloadable2234×1409 PNG pages. All
100 were downloaded, totaling14,699,482 bytes, and checked against the SHA256
values in the OSF file metadata. Original source URLs, IDs, sizes and hashes are
recorded in `data/external/classic_bongard/download_provenance.json`.

The [authors' crop utility](https://github.com/ml-research/bongard-in-wonderland/blob/main/utils/crop_images.py)
documents the fixed cell layout. Our independent preparer takes430×430 interiors
on the same480-pixel grid, starting at(9,9) for the left group and(1312,9) for the
right group. It does not run that utility. Crops are ordered by row. The final
bottom-right cell in each group is reserved as query; the other five are support.
The original pages, crop coordinates and source hashes remain auditable.

Frozen manifests:

- `data/external/classic_bongard/episodes_dev.json`: p0001–p0020.
- `data/external/classic_bongard/episodes_test.json`: p0021–p0100.
- `episodes_dev_audit.json` and `episodes_test_audit.json`: eligibility and hashes.

Problem p0001 intentionally contains repeated blank positive cells. Its query
is byte-identical to support, so strict held-out evaluation excludes that problem
and records the reason. The remaining19 development and80 test episodes pass
the exact-duplicate check. No solution descriptions were downloaded or used by
the adapters. This is a local within-problem holdout protocol, not an official
classic train/test benchmark split.

```powershell
.\.venv\Scripts\python.exe -m src.datasets.fetch_classic data/external/classic_bongard
```

```python
from pathlib import Path
from src.datasets import load_classic
root = Path("data/external/classic_bongard")
episodes = load_classic(root, manifest_file=root / "episodes_dev.json")
# Catch DatasetError from prepare() and report exclusions; do not score them.
```

The inspected OSF node metadata does not specify a redistribution license.
Images and derived crops are retained locally for this requested evaluation and
excluded from Git; no image-redistribution permission is inferred.

## Bongard-HOI

The [official repository](https://github.com/NVlabs/Bongard-HOI) describes a
natural-image benchmark of human-object interactions with difficult negatives
that share object categories. The [data-preparation instructions](https://github.com/NVlabs/Bongard-HOI/blob/master/assets/dataset.md)
link the fixed [Zenodo release7079175](https://zenodo.org/records/7079175).
Its metadata lists a50,440,335,360-byte image TAR and207,667,200-byte annotation
TAR. The50GB image archive was not downloaded.

Only bytes0–1048575 of the annotation archive were fetched. This prefix contains
complete initial episodes from the14,796,534-byte
`bongard_hoi_val_unseen_obj_unseen_act.json` member. The first episode is preserved
as an attributed parser fixture in `tests/fixtures/datasets/`; download bounds,
source member and prefix hash are recorded in
`data/external/bongard_hoi/provenance.json`. This validates actual annotation
format without inventing performance on unavailable photographs.

The [official loader](https://github.com/NVlabs/Bongard-HOI/blob/master/datasets/image_bongard_bbox.py)
reads `[positive_records, negative_records, concept]`, with seven records per
group. The first six are support and the final record is query. `load_hoi`
preserves that membership. Each image is cropped using `crop_bbox`; the official
OpenImages coordinate correction of1.6 is explicit and configurable. Subject,
object and action annotations are discarded at the pixel boundary. Annotation
crops are privileged localization information and must be disclosed as part of
this evaluation condition. No detection-box pickle is loaded.

```python
from src.datasets import load_hoi
episodes = load_hoi(
    "/local/annotations/bongard_hoi_test_seen_obj_seen_act.json",
    "/local/hake/images",
    split="test_seen_obj_seen_act",
)
```

Preserve the four official test conditions: seen/unseen objects crossed with
seen/unseen actions. Report each condition and their mean; do not replace those
splits with a random image split. Local roots follow the published HAKE directory
tree. Our loader supports all nine official train/validation/test filenames.

The [NVIDIA code license](https://github.com/NVlabs/Bongard-HOI/blob/master/LICENSE)
limits upstream code to noncommercial research or evaluation. Our parser is
independently written and does not copy that code. The Zenodo dataset record
separately declares CC BY4.0. Source photographs originate in constituent HAKE
collections; the code license is not a blanket image license.

## Bongard-LOGO

The [official project](https://github.com/NVlabs/Bongard-LOGO) provides12,000
synthetic problems: free-form strokes, basic shapes and abstract attributes.
Its full ShapeBongard_V2 archive is linked through Google Drive. It was not
downloaded or regenerated here. The repository's [MIT license](https://github.com/NVlabs/Bongard-LOGO/blob/master/LICENSE)
is separate from the HOI code license.

The [official dataset loader](https://github.com/NVlabs/Bongard-LOGO/blob/master/Bongard-LOGO_Baselines/datasets/shape_bongard_v2.py)
uses `ff`, `bd` and `hd` family directories, followed by
`images/TASK/{1,0}/*.png`. `load_logo` requires the supplied split JSON, exactly
seven images per group, and disjoint task identifiers between splits. It takes
sorted first-six supports and final queries without exposing task names or
ground-truth generation programs. The benchmark configuration uses `train`,
`val`, `test_ff`, `test_bd`, `test_hd_comb` and `test_hd_novel`.

```python
from src.datasets import load_logo
episodes = load_logo("/local/ShapeBongard_V2", split="test_hd_novel")
```

LOGO tests here check loader behavior and real published task-name conventions;
they are not evaluations on the downloaded12,000-problem corpus.

## Bongard-OpenWorld

The [official dataset card](https://huggingface.co/datasets/rujiewu/Bongard-OpenWorld)
contains610 training,200 validation and200 test episodes, each with14 images.
The [project repository](https://github.com/rujiewu/Bongard-OpenWorld) provides
original image URLs and warns that links can expire. It declares code under
Apache2.0 and data under CC BY-NC-SA4.0; source images are crawled and their
ownership is not claimed by the project. Original images remain ignored local
research data in this workspace.

Metadata is pinned to repository commit
`0462ba15f3be2f7f9a7e6cdd0d24314a1fabb5fe`. Before model inference, retrieval
selected the first five official validation and first ten test episodes in their
published order. It attempted every image in those fixed episodes, without using
concept difficulty or model accuracy to select successes.55/70 validation and
108/140 test images were retrieved and decoded; all47 failures remain in the
portable index files with source URL and reason. Images total41,610,148 bytes.
URLs are restricted to unauthenticated public HTTP(S), downloads are bounded,
and no downloaded scripts are run.

The complete official six-shot condition has zero fully available validation
episodes and two test episodes(0520,0014). For a useful preliminary local test,
the explicit availability-filtered condition instead takes the first three
available original support images per group. Both original final queries must
still exist; a failed query is never replaced by a support image. This produces
three validation and eight test episodes. The two missing-query episodes in each
cohort are recorded as exclusions. This3-shot availability condition must not be
presented as the official six-shot test accuracy or as an unbiased sample of all
OpenWorld concepts.

Frozen files are `data/external/bongard_openworld/cohort_dev.json` and
`cohort_test.json`. They retain original split membership, attempted IDs,
inclusions, exclusions, hashes and the changed shot protocol. Source filenames
embed URLs and are not portable to Windows; the image index maps exact original
references to opaque local filenames. No filename reaches a model.

```powershell
.\.venv\Scripts\python.exe -m src.datasets.fetch_openworld data/external/bongard_openworld --split val --count 5 --freeze-three-shot-cohort
.\.venv\Scripts\python.exe -m src.datasets.fetch_openworld data/external/bongard_openworld --split test --count 10 --freeze-three-shot-cohort
```

```python
from src.datasets import load_openworld_cohort
episodes = load_openworld_cohort(
    "data/external/bongard_openworld",
    cohort_file="data/external/bongard_openworld/cohort_dev.json",
)
```

The retrieval command refreshes public URL availability; it does not justify
silently changing a frozen evaluation cohort. Keep old indices/results for an
existing run. Report test results only after choosing model and fitting settings
using development data. For missing data, unknown predictions and model failures,
report coverage and denominators as well as conditional accuracy.

## Reproduce adapter checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dataset_adapters.py -q
```

These checks cover genuine HOI annotation parsing, fixed held-out membership,
pixel-only inputs, label-order randomization, malformed data, Windows path and
symlink escapes, duplicate source/copy detection, crop coordinates, LOGO split
isolation, and the OpenWorld availability condition. They do not imply that any
model has learned an arbitrary visual concept.
