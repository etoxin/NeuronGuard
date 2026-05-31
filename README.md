# NeuronGuard

**An ultra-fast, cache-aligned neuromorphic event engine bridging bare-metal Rust with an idiomatic, GIL-free Python SDK.**

NeuronGuard is not a deep learning matrix-multiplication library. It is a highly specialized associative memory engine designed for edge intelligence, real-time control, and transparent classification. It runs **entirely on the CPU**, training on 100,000+ samples in under 3 seconds, all while fitting entirely within L1/L2 CPU cache lines.

---

## The Technology (How it works)

Instead of dense floating-point matrices, NeuronGuard models intelligence as a network of **cache-aligned, thread-bounded neurons** that communicate via discrete event spikes. 

* **Cache-Aligned Memory**: Every neuron is packed into exactly 64 bytes. This guarantees deterministic hardware pre-fetching and completely eliminates false sharing.
* **GIL-Free Parallelism**: Stream and batch processing instantly drops the Python Global Interpreter Lock (GIL), utilizing all CPU cores for parallel, lock-free memory mutations.
* **Zero-Copy Serialization**: Models are flat, pointerless memory structures. Saving is an instant memory dump, and loading uses `mmap` to map the weights straight from disk into memory in `<1ms`.

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
# allowing the engine to instantly learn 2D non-linear patterns (e.g. Physics Data).
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
Because NeuronGuard uses biological topological plasticity instead of backpropagation, models can be updated on the fly without retraining from scratch.

```python
# 1. A stream of new labelled data arrives in production
correction = [(1, "my account upgrade to the premium plan failed")]

# 2. Update the model instantly (zero-overhead, <1ms)
classifier.update_records(correction)
```

### 5. Instant Machine Unlearning (GDPR Compliance)
In traditional Deep Learning, if a user requests their data be deleted (Right to be Forgotten), the entire model must often be retrained from scratch. Because NeuronGuard uses reversible Hebbian plasticity rather than entangled gradient descent, you can instantly erase a record's influence by simply passing it to `.unlearn_records()`.

```python
# Instantly subtracts the exact synaptic weight modifications caused by this record
classifier.unlearn_records([(0, "Delete my private email address test@example.com")])
```

### 6. Raw Rust Batch API (GIL-Free)
For ultimate performance, bypass the high-level classes and write batch processing loops using the raw Rust `NeuronGuardField` directly.

```python
import time
import neuronguard as ng

field = ng.NeuronGuardField(sensory_count=10000, motor_count=4)

# Train 100,000 tasks instantly across all cores using Rayon
field.train_batch(batch_train_tasks, amplify_delta=15, suppress_delta=5)

# Predict 100,000 times in milliseconds
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

# cd into the example you want to run, and run: 
mise run

# you may need to download data before you can run.

# Example of scripts you can run. Other examples are similar.
mise run examples:amazon_reviews:download_data
mise run examples:amazon_reviews:run
# 2. Semantic Generalization & Continual Learning
mise run examples:sparse_embeddings:run
mise run examples:continual_learning:run
mise run examples:melbourne_cup:run
mise run examples:team_composition:run
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
