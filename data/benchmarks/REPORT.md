# External Bongard evaluation — 18 October 2026

The project is a broader local research solver, **not a universal Bongard solver**. Measured drawing features improve over their simple global-prototype baseline. Natural-image results remain limited, and combining representations does not consistently improve performance. These measurements concern held-out image classification, not recovery of the intended human rule.

## Frozen test results

| Test condition | Representation | Forced accuracy | Global prototype | Answered / queries | Correct / answered |
|---|---|---:|---:|---:|---:|
| Classic / 5-shot | Measured features + episodic learner | 71.9% | 49.4% | 69 / 160 | 58 / 69 (84.1%) |
| Classic / 5-shot | Learned global/dense features + episodic learner | 64.4% | 61.9% | 36 / 160 | 34 / 36 (94.4%) |
| Classic / 5-shot | Combined features + episodic learner | 70.6% | 50.6% | 71 / 160 | 60 / 71 (84.5%) |
| OpenWorld / available 3-shot | Measured features + episodic learner | 50.0% | 54.1% | 14 / 74 | 8 / 14 (57.1%) |
| OpenWorld / available 3-shot | Learned global/dense features + episodic learner | 64.9% | 62.2% | 19 / 74 | 14 / 19 (73.7%) |
| OpenWorld / available 3-shot | Combined features + episodic learner | 56.8% | 62.2% | 16 / 74 | 11 / 16 (68.8%) |

**Forced accuracy** ignores the abstention gate and scores every eligible query (an exact tie earns half credit). The app normally reports unresolved decisions when empirical validation, margin, stability or novelty checks fail. High accuracy among answered images must be read together with answer coverage. Unresolved images are not correct answers.

## Protocol and denominators

- Classic: p0001–p0020 were development; p0021–p0100 were the 80 frozen test problems. Each test problem used five supports per group and the original sixth cell as the held-out query: 160 test queries. The repeated blank support/query in development problem p0001 was excluded. No classic test problem was excluded.
- OpenWorld: the first 20 official validation and first 50 official test episodes were selected before inference. Public image availability left 15 development and 37 test problems. The test results use the first three available original supports per group and the original index-6 query: 74 eligible test queries. Thirteen planned test problems were unavailable and are listed in the raw cohort metadata. This is an availability-filtered 3-shot condition, **not the official complete 6-shot benchmark**.
- Full-planned-cohort answer coverage for OpenWorld is 14% measured, 19% learned and 16% combined, including unavailable problems in the coverage denominator. Accuracy is reported only on available queries; missing downloads are not silently removed from the cohort counts.
- Feature/learner selection and normalization used support-only nested validation. Query labels were used only after prediction. Encoders were frozen; no cross-problem fine-tuning occurred. Source hashes, source-image hashes, crops, membership and configurations are preserved in the raw reports.
- The benchmark class named `structural` means numeric measurements plus episodic statistical learning. It is not the separate executable symbolic DSL solver. The old OpenCV/CNN/DSL and whole-image TensorFlow baselines remain available; their earlier synthetic pilot is a different evaluation condition.
- Public classic images/concepts may have occurred in pretrained-model data. Exact source/content overlap checks are not proof of absence of near-duplicates or pretraining contamination. Two queries in one problem are correlated; no significance claim is made for small percentage differences.

## Local language search: bounded development pilots

The Qwen3-VL 8B NF4 profile was evaluated with one hypothesis per proposal, up to two support-only repair rounds and group-blind per-image checks. These small development pilots are control-flow and capability demonstrations, not held-out language-model benchmarks.

- Classic p0002: a wrong closed-curve hypothesis passed only 6/10 support checks and was rejected. Both unseen images remained unresolved.
- Classic p0003: an open/closed hypothesis failed 7/10 support consistency. Repair proposed outline versus filled, passed 10/10 support checks and correctly classified both unseen images.
- Classic pilot total: two eligible problems, four queries, two answered and two correct; one repeated-blank problem excluded. Group-blind checks use the same supports that generated a concept and can share model errors.
- OpenWorld pilot: the first three eligible validation episodes (0448, 0371, 0114) were frozen before language inference. The first two produced no support-consistent concept. Episode 0114 passed all six support checks, but only one of its two unseen images was classified correctly. Overall: six queries, two answered, one correct. This directly demonstrates that model self-checking can still accept a wrong generalization.

## Practical limits

The semantic learner improves only two forced query decisions over its prototype baseline on this 74-query natural-image subset (48 versus 46). The combined learner does worse than the semantic learner there. The drawing learner still makes incorrect accepted decisions, and many problems remain unresolved. A larger vocabulary or a pretrained model alone does not solve unrestricted abstraction, relational composition or concept identification.

The interface therefore keeps learned decisions, language hypotheses and executable symbolic rules separate. Similarity scores are not presented as probabilities. Model-generated JSON is never executed. The dedicated supplied-arrow and Akinator regressions remain covered by the full test suite.

## Artifacts and verification

- [Classic raw test report](classic_test.json)
- [OpenWorld raw test report](openworld_test.json)
- [Classic language development pilot](language_development_pilot.json)
- [Natural-image language development pilot](language_openworld_development_pilot.json)
- [Runtime environment](environment.json)
- [Dataset access, licenses and exact protocols](../../docs/DATASETS.md)
- Local verification: **264 passed, 1 skipped**; the skip requires Windows symlink privileges. GPU model inference was checked separately. CPU GitHub CI also passed; it does not download optional model weights.

Windows checkpoint replacement intermittently failed while another reader held a file. The continuation changed only `_save` to retry transient errors. The non-writer AST was verified identical to commit `7677726`; completed records and frozen input/inference settings were preserved. Both raw test reports retain the previous configuration and this recovery audit. Final inference code is available at commit `a3d881f`.
