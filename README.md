# NeuronGuard

**A fast, transparent sparse decision layer for edge inference, streaming classification, and routing.**

NeuronGuard is a CPU-first associative classifier for workloads that value
low single-event latency, inexpensive online updates, and inspectable feature
contributions. It converts text tokens or bucketed numerical features into a
small set of class associations stored in a flat Rust memory layout.

It is not a replacement for general deep learning, vectorized linear models, or
boosted trees. Its strongest measured use case is making one transparent
decision at a time in latency-sensitive streams, edge services, and first-stage
routers.

## Measured trade-offs

On the repository's reproducible three-dataset benchmark, NeuronGuard matched or
exceeded Gaussian Naive Bayes accuracy on all three fixed test splits (fraud
0.9990 vs 0.9762, breast cancer 0.9649 vs 0.9561, wine 1.0000 vs 1.0000).
It delivered approximately 4.3 µs median single-record latency on fraud,
compared with 73 µs for logistic regression and 2,592 µs for a histogram boosted
tree through these Python APIs. Its fraud PR-AUC was 0.699, below logistic
regression at 0.719 and the boosted tree at 0.736. Optimized baselines still win
batch throughput and training time.

See [benchmark methodology and results](benchmarks/RESULTS.md) for the complete
comparison and machine details.

---

## How it works

Instead of dense floating-point matrices, each sensory token owns a bounded set
of weighted class connections. Prediction sums the connections activated by an
input and selects a class using either argmax or a tuned binary score threshold.

* **Cache-aligned payloads**: Every serialized neuron occupies one aligned 64-byte payload, improving predictable addressing and locality.
* **Safe concurrent access**: Per-neuron read/write synchronization prevents training and inference data races. Contended batch updates are serialized rather than silently discarded.
* **Request-local scoring**: Predictions calculate scores locally, so concurrent callers cannot reset or contaminate one another.
* **GIL-free native work**: Rust prediction and training operations release the Python GIL; batch operations can use Rayon across independent neurons or records.
* **Memory-mapped models**: Flat weight files can be mapped directly into memory after strict length validation.
* **Quantile tabular features**: Numerical classifiers can use equal-width or exact training-set quantile buckets and validation-tuned binary thresholds.
* **Normalized learning**: Tabular models default to smoothed categorical log-likelihood weights, avoiding order dependence and `i16` saturation from repeated updates. Legacy Hebbian learning remains available explicitly.

---

## Python SDK Reference

The high-level Python SDK handles vocabulary building, tokenization (in native Rust), hyperparameter scaling, and out-of-the-box diagnostics.

### `TextClassifier`
Handles high-speed text classification with discriminative vocabulary scoring.

```python
from neuronguard import TextClassifier

# Initialise the TextClassifier with 4 classes
classifier = TextClassifier(
    num_classes=4,
    vocab_size=1000,
    class_names=["World", "Sports", "Business", "Sci/Tech"],
)

# Train the model. Identify the text,label index in the CSV and run for 5 epochs
classifier.fit("train.csv", text_col=[1, 2], label_col=0, epochs=5)

# Evaluate with a similar setup.
accuracy, report = classifier.evaluate("test.csv", text_col=[1, 2], label_col=0)

# Make a prediction
print(classifier.predict_name("Football match ends in a draw"))  # -> "Sports"

# Save the model
classifier.save("./model_dir")

# Load the model and make a prediction
fast_model = TextClassifier.load("./model_dir")
print(fast_model.predict_name("Football match ends in a draw"))
```

### 2. `TabularClassifier`
Automatically buckets continuous numerical features and handles extreme class imbalances (like fraud detection) using internal class weighting.

```python
from neuronguard import TabularClassifier

# Features are automatically bucketed into 10 buckets each.
# use_feature_interactions=True mathematically hashes pairs of metrics together, 
# allowing the engine to represent 2D non-linear patterns.
classifier = TabularClassifier(
    num_classes=2, 
    num_features=5, 
    buckets_per_feature=10,
    use_feature_interactions=True,
    interaction_vocab_size=1000000
)

# Seamlessly handles continuous data in O(1) time
classifier.fit(
    records=[(V1, V2, V3, V4, V5, label)], 
    feature_indices=[0, 1, 2, 3, 4], 
    label_index=5
)
prediction = classifier.predict([1.2, 0.4, 9.9, 3.1, 0.0])
```

### 3. Diagnostics & Explainability (White-Box AI)
Because NeuronGuard is a direct associative memory, we can perfectly trace exactly which tokens contributed to a prediction. No more guessing why the model failed.

```python
# 1. Extract exactly what the model learned for a class
classifier.print_class_features(class_idx=1, top_k=3)
# Output:
#   - 'urgent' (Weight: +105)
#   - 'bank' (Weight: +105)

# 2. Transparently explain a single prediction
classifier.print_explanation("urgent meeting to reset password")
# Output:
#   'urgent    ' -> Spam: +105w
#   'meeting   ' -> Work: +105w
#   'reset     ' -> Spam: +105w
```

### 4. Continuous Online Learning
Sparse class associations can be updated without rebuilding the complete model.

```python
# 1. A stream of new labelled data arrives in production
correction = [(1, "my account upgrade to the premium plan failed")]

# 2. Update the existing associations
classifier.update_records(correction)
```

### 5. Experimental inverse updates
NeuronGuard can apply the inverse of a record's normal training deltas. This is
useful for correction experiments, but it is not guaranteed to erase an exact
historical influence after weight saturation, connection eviction, or later
overlapping updates, and it should not be treated as certified regulatory
machine unlearning.

```python
# Apply inverse deltas for this record
classifier.unlearn_records([(0, "Delete my private email address test@example.com")])
```

### 6. Raw Rust Batch API (GIL-Free)
For lower-level control, use the raw Rust-backed `NeuronGuardField` directly.

```python
import time
import neuronguard as ng

field = ng.NeuronGuardField(sensory_count=10000, motor_count=4)

# Train a batch using deterministic per-token update ordering
field.train_batch(batch_train_tasks, amplify_delta=15, suppress_delta=5)

# Predict a batch using Rayon
results = field.predict_batch(batch_predict_tasks)
```

---

## Running the Examples

All examples are configured cleanly via `mise` with namespaced tasks.

```bash
# 1. First, install the tools and build the extension
mise trust
mise run setup:py
mise run build:py

# Change into the example directory, then invoke its namespaced task.
cd example/fraud_scanner
mise run examples:fraud_scanner:download_data
mise run examples:fraud_scanner:run

# you may need to download data before you can run.

# Other examples follow the same directory-local pattern.
```

## Installing 

```bash
pip install neuronguard
```

[https://pypi.org/project/neuronguard/](https://pypi.org/project/neuronguard/)

## Building Locally

This project uses [mise](https://mise.jdx.dev/) and [uv](https://github.com/astral-sh/uv) to manage toolchains.

```bash
# 1. Install toolchains
mise install

# 2. Build the Rust extension using maturin
mise run build:py

# 3. (Optional) Run Rust and Python verification tests
mise run test:rust
mise run test:py
```

## License
Apache License 2.0
