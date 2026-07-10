# Benchmark results

Measured on an Apple M5 Pro (18 cores, 48 GB RAM), macOS 26.5.1, Python
3.11.15, Rust 1.96.0, scikit-learn 1.9.0, and NumPy 2.4.6.

| Dataset | Model | Accuracy | PR-AUC | Train (s) | Batch µs/sample | Single p50 (µs) | Peak train MB | Model KB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| credit-card-fraud | NeuronGuard | 0.9990 | 0.5721 | 2.5795 | 4.02 | 4.04 | 388.28 | 38.79 |
| credit-card-fraud | Naive Bayes | 0.9762 | 0.0797 | 0.0175 | 0.17 | 26.50 | 37.91 | 1.45 |
| credit-card-fraud | Logistic regression | 0.9748 | 0.7194 | 0.0976 | 0.08 | 67.58 | 0.97 | 1.96 |
| credit-card-fraud | Boosted tree | 0.9981 | 0.7361 | 3.6864 | 0.15 | 656.69 | 222.11 | 181.70 |
| breast-cancer | NeuronGuard | 0.9298 | 0.9911 | 0.0030 | 4.14 | 3.92 | 1.11 | 37.55 |
| breast-cancer | Naive Bayes | 0.9561 | 0.9933 | 0.0003 | 1.50 | 26.75 | 0.08 | 1.48 |
| breast-cancer | Logistic regression | 0.9561 | 0.9963 | 0.0022 | 2.57 | 68.15 | 0.56 | 1.99 |
| breast-cancer | Boosted tree | 0.9474 | 0.9940 | 1.2040 | 29.83 | 1838.56 | 5.14 | 220.76 |
| wine | NeuronGuard | 0.8333 | 0.9967 | 0.0004 | 3.28 | 2.00 | 0.17 | 16.06 |
| wine | Naive Bayes | 1.0000 | 1.0000 | 0.0004 | 4.58 | 30.67 | 0.08 | 1.18 |
| wine | Logistic regression | 0.9722 | 1.0000 | 0.0026 | 7.04 | 65.92 | 0.55 | 1.69 |
| wine | Boosted tree | 0.9722 | 1.0000 | 0.1340 | 316.13 | 5397.38 | 2.30 | 148.23 |

## Interpretation

NeuronGuard's validated advantage is single-record latency. It was approximately
17 times faster than logistic regression and 163 times faster than the boosted
tree for an individual fraud transaction through these Python APIs. It also
retained strong ranking quality on breast cancer and wine.

It did not outperform optimized vectorized baselines for batch inference,
training time, memory, model size, or predictive quality. On fraud, its PR-AUC
of 0.5721 trailed logistic regression at 0.7194 and boosted trees at 0.7361.
The exact quantile implementation currently materializes Python records and
per-feature value arrays, producing a high peak training RSS on the fraud data.

These results support positioning NeuronGuard as a transparent, low-latency
per-event decision layer with inexpensive online updates—not as a universal
replacement for vectorized statistical or tree models.

## Methodology notes

- Deterministic stratified 60/20/20 train/validation/test splits.
- NeuronGuard used 16 exact quantile buckets per feature and validation-tuned
  binary F1 thresholds.
- Logistic regression and boosted trees used balanced class weights.
- Multi-class PR-AUC is macro one-vs-rest average precision.
- Batch latency processes the complete test set; single latency is the median of
  up to 500 one-record calls after warm-up.
- Peak RSS is the maximum process RSS increase during model fitting after the
  dataset is loaded. Native allocators and sampling intervals make it approximate.
- Serialized size includes NeuronGuard weights and JSON configuration, or the
  highest-protocol pickle representation for baselines.
- Results are machine-specific and should be regenerated after material changes.
