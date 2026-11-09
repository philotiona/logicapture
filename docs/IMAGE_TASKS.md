# Sort and inspect your own images

Open **Sort images** in Logicapture. This workspace learns a two-group distinction from your examples and applies the same fitted task to subsequent batches. It uses the main solver's visual representations, reference cross-validation and abstention checks.

## Use a task

1. Name the task and its two groups, such as two product categories. Choose **Photographs and screenshots** for everyday images or **Black-and-white drawings** for diagram inspection.
2. Upload 4–6 distinct, varied reference images per group. ZIPs are accepted too; 2–12 references per group are supported. Give examples of both outcomes and vary backgrounds, lighting and viewpoints that should not determine the group. Review the displayed groups and press **Learn from these examples**.
3. To reuse the groups later, open **Save this setup for later** and press **Save sorting setup**. The ZIP contains the reference images and settings. Reopen it with **Open saved setup** and learn from those references again using the current local solver. A saved setup is not a pickled model or a promise of identical predictions after software changes.
4. Upload new images or a ZIP and press **Sort these images**. Up to 100 files and 100 MB of expanded input are supported. ZIPs containing unsupported non-image files list those as skipped; corrupt images remain visible as errors. Archives are read in memory without extracting paths.
5. Review the decisions. The preview shows the selected image and its closest reference examples for feature-based tasks. A tentative suggestion may be shown when automatic checks fail; it remains in **Needs review** until you explicitly label it. Select a group or **Needs review**, then press **Apply label**. You can also return to the original automatic decision.
6. Press **Download sorted images + report**. The ZIP contains the original encoded files, sorted folders, `results.csv`, `results.json`, and `task.zip` for reuse. Distinct numbered output names preserve files with the same basename. Image bytes, metadata and original resolution are retained.

Group names label the interface and exports; they never enter visual feature extraction or rule fitting. No new image batch retrains the task. Changing reference pixels, group names, task name or representation invalidates the previous fit in the interface; changing a batch invalidates its previous results. Fitting again clears the earlier batch decisions.

## Understand decisions and review

The drawing route first attempts the existing executable rules; otherwise it uses measured drawing features. The photo route uses the installed frozen SigLIP encoder and the existing support-adapted classifier. Missing photo weights produce an installation message instead of triggering an implicit download. The base application remains usable without optional model dependencies.

Automatic feature decisions require adequate reference cross-validation, a sufficient decision margin, agreement across reference refits and proximity to the reference examples. These checks can abstain and can still accept a wrong answer. Scores are not class probabilities. Closest examples and tentative suggestions are review aids, not proof of membership.

Repeated reference pixels within a group and conflicting labels across groups are rejected. A new image that matches a normalized reference is sent to review instead of being counted as independent evidence. Human review changes export labels only: the original model decision, reasons and the fact of human intervention remain recorded. Unreadable inputs stay in the `errors` folder even if a review label is supplied.

Inputs are checked for dimensions and animation. Analysis normalizes EXIF orientation, composites transparency over white and scales images to at most 1280 pixels on the longest side; the shortest side must still contain at least three pixels. A 24-megapixel source-image limit and 60-million-pixel decoded analysis budget bound batch memory use. Reference groups together are limited to 30 million analysis pixels so their lossless saved task stays within the import budget. Downloads contain the untouched original input bytes. Upload controls allow files up to 100 MB, with the same 100 MB aggregate and expanded-size checks in the backend.

## Verification and limits

Automated tests exercise task save/reopen, reference contradictions, unsafe and duplicate ZIP members, invalid images, frozen fitting across batches, reference-copy handling, human correction, stale-input invalidation and original-byte preservation. The complete app flow is covered through Streamlit's test runner.

An actual local SigLIP run is retained in [the photo workflow report](../data/regressions/image_tasks_photo_pilot.json). It includes two separate development checks:

| Check | References | New images | Automatically correct | Wrong automatic | Needs review |
|---|---|---:|---:|---:|---:|
| Penguins / tortoises category sorting | 4 per category | 3 | 3 | 0 | 0 |
| First five available frozen OpenWorld development episodes | 3 per group | 10 | 1 | 0 | 9 |

Every exported original was verified byte-for-byte. The category task was chosen by its category names before category inference; its remaining available annotated source images were withheld. The three category queries include illustrations as well as photographic imagery, and correctness uses the source category annotations. This is a small functional demonstration, not measured accuracy for other categories. Context or pretraining overlap may help. The broader abstraction check exposes the existing solver's limited automatic coverage. Neither check uses the user's personal collection, and no improvement to general recognition accuracy is claimed.

The local images and model weights remain untracked. Reproduce the photo checks after preparing the existing expanded development cohort and installing the optional local photo model:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.image_tasks_pilot
```

The task fits once per explicit action in the current session. Reopening after an app restart requires fitting the saved reference ZIP. Corrections are retained in the downloadable results; they do not automatically become new training examples. For a better task, add independently collected examples to its reference groups and fit again, then evaluate on a new batch.
