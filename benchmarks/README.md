# Model benchmarks

This benchmark compares NeuronGuard with Gaussian Naive Bayes, balanced logistic
regression, and a balanced histogram gradient-boosted tree on three tabular
datasets. It reports accuracy, precision-recall AUC, training time, batch and
single-record inference latency, peak training RSS above the loaded-dataset
baseline, and serialized model size.

The datasets are the complete credit-card fraud dataset used by the fraud
example, scikit-learn breast cancer, and scikit-learn wine. Every model receives
the same deterministic stratified 60/20/20 train/validation/test split.
NeuronGuard uses 16 exact quantile buckets per feature and tunes its binary
decision threshold on the validation split. Baselines train on the same training
split; logistic regression and boosted trees use balanced class weights.

From the repository root:

```bash
mise run setup:py
mise run build:py
mise run benchmark
```

Run the fraud example's download task first if its complete dataset is absent.
Workers run in separate processes so memory from earlier models does not affect
later peak-RSS measurements. Timings remain machine-specific and should always
be published with hardware and software details.
