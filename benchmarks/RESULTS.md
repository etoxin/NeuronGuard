# Benchmark results

Measured on an Apple M5 Pro (18 cores, 48 GB RAM), macOS 26.5.1, Python
3.11.15, Rust 1.96.0, scikit-learn 1.9.0, and NumPy 2.4.6.

| Dataset | Model | Accuracy | PR-AUC | Train (s) | Batch µs/sample | Single p50 (µs) | Peak train MB | Model KB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| credit-card-fraud | NeuronGuard | 0.9992 | 0.6087 | 1.9886 | 4.04 | 4.00 | 14.62 | 40.98 |
| credit-card-fraud | Naive Bayes | 0.9762 | 0.0797 | 0.0211 | 0.20 | 28.83 | 37.84 | 1.45 |
| credit-card-fraud | Logistic regression | 0.9748 | 0.7194 | 0.0924 | 0.09 | 68.71 | 0.95 | 1.96 |
| credit-card-fraud | Boosted tree | 0.9981 | 0.7361 | 3.6839 | 0.15 | 634.75 | 218.67 | 181.70 |
| breast-cancer | NeuronGuard | 0.9211 | 0.9929 | 0.0041 | 4.22 | 3.96 | 0.12 | 39.48 |
| breast-cancer | Naive Bayes | 0.9561 | 0.9933 | 0.0004 | 1.50 | 26.65 | 0.11 | 1.48 |
| breast-cancer | Logistic regression | 0.9561 | 0.9963 | 0.0022 | 2.43 | 67.96 | 0.52 | 1.99 |
| breast-cancer | Boosted tree | 0.9474 | 0.9940 | 1.0274 | 32.22 | 1791.27 | 5.23 | 220.76 |
| wine | NeuronGuard | 0.9444 | 1.0000 | 0.0006 | 6.71 | 1.92 | 0.08 | 17.13 |
| wine | Naive Bayes | 1.0000 | 1.0000 | 0.0003 | 4.60 | 31.21 | 0.11 | 1.18 |
| wine | Logistic regression | 0.9722 | 1.0000 | 0.0021 | 6.28 | 65.79 | 0.52 | 1.69 |
| wine | Boosted tree | 0.9722 | 1.0000 | 0.1329 | 292.33 | 5334.60 | 2.38 | 148.23 |

## Interpretation

NeuronGuard's validated advantage is single-record latency. It was approximately
17 times faster than logistic regression and 159 times faster than the boosted
tree for an individual fraud transaction through these Python APIs. The
normalized likelihood learner raised fraud PR-AUC from the previous 0.5721 to
0.6087 and wine accuracy from 0.8333 to 0.9444.

It did not outperform optimized vectorized baselines for batch inference,
training time, memory, model size, or predictive quality. On fraud, its PR-AUC
of 0.6087 trailed logistic regression at 0.7194 and boosted trees at 0.7361.
The array fitting path and one-column-at-a-time exact quantiles reduced fraud
training peak RSS from the previous 388 MB to approximately 15 MB.

These results support positioning NeuronGuard as a transparent, low-latency
per-event decision layer with inexpensive online updates—not as a universal
replacement for vectorized statistical or tree models.

## Methodology notes

- Deterministic stratified 60/20/20 train/validation/test splits.
- NeuronGuard used 16 exact quantile buckets per feature, smoothed categorical
  log-likelihood weights, and validation-tuned binary F1 thresholds.
- Logistic regression and boosted trees used balanced class weights.
- Multi-class PR-AUC is macro one-vs-rest average precision.
- Batch latency processes the complete test set; single latency is the median of
  up to 500 one-record calls after warm-up.
- Peak RSS is the maximum process RSS increase during model fitting after the
  dataset is loaded. Native allocators and sampling intervals make it approximate.
- Serialized size includes NeuronGuard weights and JSON configuration, or the
  highest-protocol pickle representation for baselines.
- Results are machine-specific and should be regenerated after material changes.
