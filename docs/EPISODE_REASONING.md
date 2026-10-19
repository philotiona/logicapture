# Learned episode prediction

The new `src/reasoning` package learns a binary decision boundary from the current positive and negative image supports. It receives numeric representations, not shape names, a list of candidate concepts, or complete rules. Its predictions are **not executable symbolic rules**. The earlier structural DSL remains a separate system.

## Why this architecture

The primary paper [Support-Set Context Matters for Bongard Problems](https://arxiv.org/html/2309.03468v2) motivates adapting features to the complete support set. It finds that support-derived feature standardization improves simple prototype and SVM classifiers. Its reported photo and abstract-drawing results use different representation training; consequently, a photo encoder's performance must not be assumed to transfer to geometric puzzles.

[SigLIP 2](https://arxiv.org/abs/2502.14786) offers image-language representations with dense-feature objectives. The official [base model card](https://huggingface.co/google/siglip2-base-patch16-224) provides a practical pretrained source for this application's semantic encoder. This implementation keeps its reasoning interface independent of that model. [DINOv2](https://github.com/facebookresearch/dinov2) and [CLIP](https://github.com/openai/CLIP) can supply alternative representations through the same contract; no checkpoint is downloaded by the reasoning package.

The [Bongard-HOI benchmark](https://github.com/NVlabs/Bongard-HOI) uses natural images and hard negatives that differ in their interaction. Recognizing an object category alone is insufficient. Benchmark accuracy is query classification accuracy, rather than proof that the intended linguistic concept has been recovered.

## Interface

```python
from src.reasoning import FeatureBatch, EpisodeSupport, EpisodeClassifier

# Arrays come from a frozen visual encoder. No query labels enter these calls.
support = EpisodeSupport(support_features, tuple(support_labels), episode_id="example")
fitted = EpisodeClassifier().fit(support)
predictions = fitted.predict(query_features)
```

`FeatureBatch` contains unique image IDs, global vectors, optional patch vectors, an encoder identity, metadata, and optional named feature views. `subset` and `concatenate` preserve row identity. `fuse_feature_batches` aligns images by ID and retains each encoder's views alongside equally normalized global blocks.

Patch views include mean, mean/deviation, and four spatial cells for square token grids. Numeric structural descriptors can supply other views. `metadata["automatic_feature_views"]` restricts automatic selection; diagnostic ablations remain accessible without entering the search. Fusion propagates those restrictions.

## Fitting and validation

The compact bank is fixed at eight candidates per automatic view:

- Cosine prototype, regularized linear ridge, and RBF kernel ridge.
- Support-standardized Euclidean prototype, ridge, and RBF kernel ridge.
- Support-contrast prototype and ridge with a maximum of sixteen selected dimensions.

The contrast transform uses regularized class differences to weight dimensions. Every mean, deviation, feature weight, selected dimension, kernel bandwidth, and fitted coefficient is computed inside the applicable training fold. Standardized quantitative measurements retain their magnitude. Normalizing every row to unit length would incorrectly turn all positive scalar values into one point.

Selection holds one support from each group out together. The reported outer leave-one-out estimate repeats this selection without the held-out support. With only two supports per group, an outer fold cannot perform inner selection; it uses the fixed first candidate. One support per group can be fitted, but has no validation estimate and causes abstention. The optional thorough profile has twelve candidates per view and must be selected before inspecting final benchmark results.

Queries never contribute normalization statistics or learner selection. Encoding supports and queries together is valid only if the encoder runs in evaluation mode and all preprocessing remains independent per image.

## Scores and abstention

Scores are signed decision evidence, normalized using training-score magnitude. **They are not probabilities.** `probability` is deliberately `None`. Tiny support sets do not justify calibrated percentages or a distribution-free correctness claim.

The default policy abstains for contradictory identical representations, absent or weak nested support validation, a small margin, disagreement among outer-fold fits, or a query outside the selected representation's support neighborhood. Retrieval shows the nearest supports in that representation. These diagnostics are not an explanation of a recovered concept.

Novelty uses cosine distance for cosine views and dimension-scaled Euclidean distance for standardized views. Its cutoff is derived only from support distances. All policy thresholds and the candidate profile are exported in diagnostics. Neither rule recovery nor probability calibration is claimed.

## Evaluation conditions

Keep development and final episodes separate. Freeze encoders, candidate bank, thresholds, and preprocessing before final evaluation. Fit on supports only; supply query membership only to evaluation. Reject physical query/support duplicates and record the split, episode IDs, encoder revision, and preprocessing.

Compare the selected classifier with a fixed global cosine nearest-prototype baseline on the **same representations and episode queries**. Report forced prediction accuracy, abstention coverage, accuracy among answered queries, and accuracy counting abstentions as unanswered. Do not compare selective accuracy alone with a baseline that answers every query. Exact decision ties receive half credit in the forced accuracy diagnostic, rather than an arbitrary preference for one group.

Compare the previous structural pipeline on the same image episodes separately, reporting its unknown assessments and its bounded rule vocabulary. Query classification and interpretable rule recovery are distinct outcomes. Bongard-HOI should also retain its official seen/unseen object/action split breakdown.

`tests/test_episode_reasoning.py` tests nuisance suppression, nonlinear discrimination, spatial evidence, scalar-magnitude preservation, inversion/permutation invariance, query isolation, contradiction handling, small supports, and ablation boundaries. These are tests of the learning implementation on controlled numeric vectors. They are **not measured pretrained-vision accuracy** on Bongard-HOI or classic Bongard pages.
