# NeuronGuard Python Extension

`neuronguard` is a high-performance, cache-aligned neuromorphic bridge that wraps a bare-metal Rust spiking neural network (SNN) core into an idiomatic Python library. It drops the Global Interpreter Lock (GIL) to execute parallel, lock-free address-mapping mutations on 64-byte `ThreadBoundedNeuron` CPU cache lines.

---

## Installation

This project uses [mise](https://mise.jdx.dev/) and [uv](https://github.com/astral-sh/uv) to manage toolchains and tasks.

```bash
# 1. Install all tools (Rust, Python, uv)
mise install

# 2. Set up the virtual environment
mise run py-setup

# 3. Build and install the Python extension
mise run py-build
```

---

## Minimal Usage Example

```python
import neuronguard as ng
import time

# Initialize Local Biological Cortex (1000 Sensory -> 4 Motor Neurons)
# Fits entirely within CPU L1/L2 cache lines (~62 KB)
cortex = ng.NeuronGuardField(sensory_count=1000, motor_count=4)

# Process incoming sensory stimuli tokens (drops GIL instantly)
active_stimuli = [42, 108, 512]
triggered_motor_id = cortex.process_stream(active_stimuli, training_mode=True)

print(f"Stimuli {active_stimuli} ➔ Triggered Motor Neuron: {triggered_motor_id}")

# Run background metabolic decay loop
cortex.tick_decay(decay_factor=0.90)
```

---

## Performance & Benchmarks

| Dataset / Task | Samples | Classes | Sensory Neurons | Training Time | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AG News Dataset** | 120,000 | 4 | 1,000 | **1.28s** | **82.14%** |
| **DBpedia Ontology Dataset** | 560,000 | 14 | 5,000 | **0.64s** | **86.26%** |
| **Credit Card Fraud Scanner** | 29,222 | 2 | 50 | **0.025s (25ms)** | **99.93%** (75% Precision) |

*Benchmarks run on an Apple M2 Pro CPU using Python 3.12.*

---

## Tasks

You can run all library and extension tasks cleanly using `mise`:

```bash
# Run the Getting Started Guide
mise run getting_started

# Run the Advanced Multi-Threaded Simulation
mise run advanced

# Run the High-Frequency Financial Fraud Scanner
mise run fraud_scanner

# Run the AG News Classifier
mise run ag_news

# Run the DBpedia Classifier & Router
mise run dbpedia

# Run the PyTorch vs NeuronGuard DBpedia Benchmark
mise run dbpedia_benchmark

# Run the Real Amazon 3,000,000 Ingestion Benchmark
mise run giant_dataset

# Run the PyTorch vs NeuronGuard 3M Giant Benchmark
mise run giant_dataset_benchmark

# Run the Pure Python Static Expression & Gesture Cloner
mise run expression_cloner

# Run Rust unit tests
mise run test

# Build and install the Python extension
mise run py-build

# Run the Python verification script
mise run py-test
```

---

## License

This project is licensed under the **Apache License 2.0**.
