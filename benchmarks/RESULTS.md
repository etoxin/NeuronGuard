# Benchmark results

Measured on an Apple M5 Pro (18 cores, 48 GB RAM), macOS 26.5.1, Python
3.11.15, Rust 1.96.0, scikit-learn 1.9.0, and NumPy 2.4.6.

| Dataset | Model | Accuracy | PR-AUC | Train (s) | Batch µs/sample | Single p50 (µs) | Peak train MB | Model KB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| credit-card-fraud | NeuronGuard | 0.9990 | 0.6987 | 1.8837 | 4.37 | 4.29 | 5.36 | 20.07 |
| credit-card-fraud | Gaussian Naive Bayes | 0.9762 | 0.0797 | 0.0181 | 0.19 | 28.21 | 37.84 | 1.45 |
| credit-card-fraud | Logistic regression | 0.9748 | 0.7194 | 0.0996 | 0.09 | 73.21 | 0.97 | 1.96 |
| credit-card-fraud | Boosted tree | 0.9981 | 0.7361 | 4.2004 | 0.26 | 2591.54 | 236.14 | 181.70 |
| breast-cancer | NeuronGuard | 0.9649 | 0.9917 | 0.0042 | 4.80 | 4.60 | 0.05 | 20.12 |
| breast-cancer | Gaussian Naive Bayes | 0.9561 | 0.9933 | 0.0004 | 1.63 | 28.46 | 0.11 | 1.48 |
| breast-cancer | Logistic regression | 0.9561 | 0.9963 | 0.0023 | 2.52 | 71.79 | 0.47 | 1.99 |
| breast-cancer | Boosted tree | 0.9474 | 0.9940 | 1.2408 | 89.23 | 5163.29 | 4.91 | 220.76 |
| wine | NeuronGuard | 1.0000 | 1.0000 | 0.0006 | 6.90 | 2.21 | 0.08 | 9.12 |
| wine | Gaussian Naive Bayes | 1.0000 | 1.0000 | 0.0004 | 4.96 | 33.15 | 0.11 | 1.18 |
| wine | Logistic regression | 0.9722 | 1.0000 | 0.0023 | 7.16 | 69.38 | 0.53 | 1.69 |
| wine | Boosted tree | 0.9722 | 1.0000 | 0.3196 | 964.74 | 17892.52 | 2.27 | 148.23 |

## Interpretation

The accuracy gap to Gaussian Naive Bayes is closed on this deterministic split:
NeuronGuard scores 0.9990 vs 0.9762 on fraud, 0.9649 vs 0.9561 on breast cancer,
and 1.0000 vs 1.0000 on wine. The fraud PR-AUC also rises from the previous
0.6087 to 0.6987, while remaining below the stronger logistic/tree ranking
baselines. The improvement comes from a pre-declared eight-bin uniform
histogram with light (0.01) likelihood smoothing, which avoids wasting most
of the small wine and breast-cancer training sets on empty 16-bin quantiles.

NeuronGuard retains its single-record latency advantage: it is approximately
17 times faster than logistic regression for an individual fraud transaction
through these Python APIs.

It still does not outperform optimized vectorized baselines for batch inference
or training time. On fraud, its PR-AUC of 0.6987 trails logistic regression at
0.7194 and boosted trees at 0.7361, while its peak training RSS is 5.36 MB.

These results support positioning NeuronGuard as a transparent, low-latency
per-event decision layer with inexpensive online updates. They do not establish
universal superiority over vectorized statistical or tree models; the split is
small for wine and breast cancer, and repeated cross-validation remains the
appropriate release gate.

## Methodology notes

- Deterministic stratified 60/20/20 train/validation/test splits.
- NeuronGuard used 8 uniform buckets per feature, smoothing=0.01, smoothed
  categorical log-likelihood weights, and validation-tuned binary accuracy
  thresholds. The fixed split is a development benchmark; repeated nested
  cross-validation is still required for release-level generalization claims.
- Logistic regression and boosted trees used balanced class weights.
- Multi-class PR-AUC is macro one-vs-rest average precision.
- Batch latency processes the complete test set; single latency is the median of
  up to 500 one-record calls after warm-up.
- Peak RSS is the maximum process RSS increase during model fitting after the
  dataset is loaded. Native allocators and sampling intervals make it approximate.
- Serialized size includes NeuronGuard weights and JSON configuration, or the
  highest-protocol pickle representation for baselines.
- Results are machine-specific and should be regenerated after material changes.
