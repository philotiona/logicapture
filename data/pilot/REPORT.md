# Measured synthetic pilot

Seed: 42. 3 problems per target, six target programs. Two panels held out per problem.

| System / condition | Held-out accuracy |
|---|---:|
| neural_perception_hard_symbolic/clean | 91.7% |
| neural_perception_hard_symbolic/noise | 91.7% |
| neural_perception_hard_symbolic/rotation | 88.9% |
| pixel_prototype/clean | 58.3% |
| pixel_prototype/noise | 58.3% |
| pixel_prototype/rotation | 61.1% |
| symbolic_cv/clean | 91.7% |
| symbolic_cv/noise | 91.7% |
| symbolic_cv/rotation | 91.7% |
| symbolic_cv_regularized/clean | 91.7% |
| symbolic_cv_regularized/noise | 91.7% |
| symbolic_cv_regularized/rotation | 91.7% |
| symbolic_ground_truth/clean | 91.7% |
| tensorflow_prototype/clean | 58.3% |
| tensorflow_prototype/noise | 52.8% |
| tensorflow_prototype/rotation | 58.3% |

Neural baseline: fresh CNN trained on 180 episodes of the first three rule families; 78544 parameters.

Rotation means +18° per object around its unchanged center. Noise means independent Gaussian RGB noise with standard deviation 18/255. They are separate conditions.

Object perception CNN: data/models/attributes.keras. It receives isolated OpenCV regions, retains softmax distributions and supplies argmax attributes to the hard symbolic evaluator. Every segmented object is classified in this benchmark.

Soft-symbolic evaluation is planned and has not been measured.

Active questions: 12.00 mean; capped runs: 0.
Random questions: 34.44 mean; capped runs: 12.

## Program recovery on clean problems

| System | Exact AST | Agreement on 100 semantic probes |
|---|---:|---:|
| symbolic_ground_truth | 50.0% | 55.6% |
| symbolic_cv | 50.0% | 55.6% |
| neural_perception_hard_symbolic | 44.4% | 50.0% |

## Perception

| Pipeline / condition | Count correct | Object recall | Shape + fill recall | Relation recall |
|---|---:|---:|---:|---:|
| neural/clean | 99.5% | 99.6% | 92.8% | 96.8% |
| neural/noise | 99.5% | 99.6% | 92.6% | 96.8% |
| neural/rotation | 99.5% | 99.6% | 93.5% | 96.9% |
| opencv/clean | 99.5% | 99.6% | 97.4% | 96.8% |
| opencv/noise | 99.5% | 99.6% | 97.4% | 96.8% |
| opencv/rotation | 99.5% | 99.6% | 97.0% | 96.9% |

The regularized CV rows use accuracy − 0.015 × AST complexity; exact rows rank by consistency first. Simple and composed target results are recorded separately by target_index and depth in results.json.

## Limits

- Small synthetic pilot; confidence intervals require a larger sample.
- Semantic recovery means agreement on 100 probes, not logical equivalence.
- Real hand-drawn evaluation has not been performed.
- Training data may admit alternative rules; perfect_hypotheses records ambiguity.
- Active and random stop at one behavior on a finite query pool, or 40 questions.

## Observed failures

27 failures recorded; inspect results.json for the exact programs and conditions.