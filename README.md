# NeuronGuard: Cache-Aligned Neuromorphic Event Engine

`neuronguard` is an ultra-high-performance, cache-aligned neuromorphic bridge that wraps a bare-metal Rust spiking neural network (SNN) core into an idiomatic, GIL-free Python library. 

By aligning all core structures to 64-byte boundaries, `neuronguard` eliminates false sharing and guarantees deterministic hardware pre-fetching. It drops the Python Global Interpreter Lock (GIL) instantly during stream processing to execute parallel, lock-free address-mapping mutations on `ThreadBoundedNeuron` CPU cache lines.

---

## Key Architectural Pillars

1. **Cache-Aligned Memory Layout**: All neural structures (`ThreadBoundedNeuron`, `PermanentNeuromorphicLine`) are spatially aligned to exactly 64 bytes, matching standard CPU cache lines to prevent cache thrashing and maximize L1/L2 cache hit rates.
2. **GIL-Free Parallelism**: Stream processing drops the Python GIL instantly, enabling background worker threads to execute lock-free, concurrent address-mapping updates in parallel.
3. **Guard/Lease Transactional Pattern**: Implements transactional, lock-free leases on specific memory addresses for safe concurrent mutations and topological plasticity without global locks.
4. **Pointerless Serialization**: Supports instant binary serialization and deserialization (< 1ms) of model weights directly to/from disk.
5. **Spatiotemporal Ensemble Mesh**: Features a flat, contiguous memory mesh with recurrent loopback feedback loops (Central Pattern Generators) for rhythmic coordination and self-stabilizing gaits.

---

## Why NeuronGuard? (Generalized Use Cases)

`neuronguard` is not a general-purpose deep learning framework like PyTorch or TensorFlow. Instead, it is a highly specialized, bare-metal neuromorphic event engine designed for edge intelligence, real-time control, and high-frequency event processing.

### 1. High-Throughput Machine Learning & Classification
* **The Challenge**: Traditional machine learning libraries (like PyTorch or scikit-learn) have heavy runtime overheads, large memory footprints, and require GPU acceleration to process high-volume data streams quickly.
* **The NeuronGuard Solution**: `neuronguard` maps inputs directly to cache-aligned neuromorphic structures. It performs classification (text, tabular, or sensor data) in **microseconds** on a single CPU core, fitting entirely within CPU L1/L2 cache lines (~62 KB) and achieving over 80-99% accuracy on standard datasets.

### 2. High-Frequency Event Processing & Stream Ingestion
* **The Challenge**: Processing millions of continuous event streams, telemetry, or network packets in Python is heavily bottlenecked by the Global Interpreter Lock (GIL) and lock contention.
* **The NeuronGuard Solution**: `neuronguard` drops the Python GIL instantly during stream processing. This allows multiple concurrent Python threads to ingest and evaluate high-velocity streams in parallel across background worker threads with zero lock contention.

### 3. Real-Time Edge Robotics & Control Systems
* **The Challenge**: Robotic coordination (like walking gaits or motor control) requires continuous, low-power feedback loops. Traditional control theory is rigid, while deep reinforcement learning is too computationally heavy for edge microcontrollers.
* **The NeuronGuard Solution**: The `SpatiotemporalEnsembleMesh` implements biological Central Pattern Generators (CPGs) that resonate rhythmically to coordinate motors. It supports real-time transactional re-patching (Guard/Lease) to instantly self-stabilize when physical perturbations (like slips or pushes) are detected.

### 4. On-Device Continuous Learning & Adaptive Systems
* **The Challenge**: Training models at the edge is nearly impossible due to the memory and compute overhead of backpropagation and gradient storage.
* **The NeuronGuard Solution**: `neuronguard` uses biological topological plasticity (Hebbian-style updates). It learns on the fly using simple integer additions and subtractions, allowing models to continuously adapt to new patterns directly on low-power edge devices without cloud connectivity.

### 5. Instant Hot-Reloading & Zero-Overhead Serialization
* **The Challenge**: Hot-reloading models in production usually requires loading massive weights files, parsing complex formats (safetensors/pickle), and rebuilding computational graphs, taking seconds or minutes.
* **The NeuronGuard Solution**: `neuronguard` weights are flat, pointerless binary arrays. The entire model state can be saved or hot-reloaded in **under 1 millisecond**, enabling instant, zero-downtime production updates.

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

## API Reference

### 1. `NeuronGuardField`
The primary neuromorphic cortex class exposed to Python. It manages sensory-to-motor connections and processes incoming stimuli streams.

#### Constructor
* **`NeuronGuardField(sensory_count: int, motor_count: int)`**
  * **`sensory_count`**: Number of sensory neurons (input vocabulary/features).
  * **`motor_count`**: Number of motor neurons (output categories/experts).
  * *Allocates a flat, contiguous block of memory where each sensory neuron is aligned to 64-byte CPU cache lines.*

#### Methods
* **`process_stream(sensory_tokens: list[int], training_mode: bool) -> int`**
  * Processes a stream of active sensory stimuli token IDs. Drops the GIL to run parallel evaluations across background worker threads.
  * **`sensory_tokens`**: List of active sensory neuron indices.
  * **`training_mode`**: If `True`, applies the Guard/Lease transactional feedback loop (amplifies winning pathway by `+15`, suppresses competing pathways by `-5`).
  * **Returns**: The index of the motor neuron with the highest accumulated potential.
* **`process_stream_sync(sensory_tokens: list[int]) -> int`**
  * Evaluates active sensory tokens synchronously on the calling thread. Extremely fast, ideal for batch evaluation or single-threaded loops.
  * **`sensory_tokens`**: List of active sensory neuron indices.
  * **Returns**: The index of the motor neuron with the highest accumulated potential.
* **`train_stream(sensory_tokens: list[int], correct_motor_id: int, amplify_delta: int, suppress_delta: int) -> None`**
  * Trains active sensory tokens to target a specific correct motor neuron.
  * **`sensory_tokens`**: List of active sensory neuron indices to train.
  * **`correct_motor_id`**: The target motor neuron index.
  * **`amplify_delta`**: Weight increment for the correct pathway (e.g., `30`).
  * **`suppress_delta`**: Weight decrement for incorrect/competing pathways (e.g., `10`).
* **`tick_decay(decay_factor: float) -> None`**
  * Simulates metabolic forgetting by shaving off a percentage of the accumulated potentials.
  * **`decay_factor`**: Multiplier applied to all motor potentials (e.g., `0.90` for 10% decay).
* **`reset_potentials() -> None`**
  * Resets all motor neuron potentials to zero.
* **`get_potentials() -> list[int]`**
  * Returns the current accumulated potentials of all motor neurons.
* **`save_weights(path: str) -> None`**
  * Serializes and saves the sensory neurons' connections to a binary file.
* **`load_weights(path: str) -> None`**
  * Loads and deserializes the sensory neurons' connections from a binary file in < 1ms.

---

### 2. `PyPermanentSpatiotemporalEnsembleMesh`
A flat, contiguous block of memory representing a spatiotemporal ensemble mesh with recurrent loopback feedback loops.

#### Constructor
* **`PyPermanentSpatiotemporalEnsembleMesh(size_per_field: int)`**
  * **`size_per_field`**: Number of lines per semantic field (total size is `size_per_field * 3`).

#### Methods
* **`process_frame(tokens: list[int], training_mode: bool, correct_target: int | None = None, decay_factor: float = 0.9) -> None`**
  * Processes a single frame pass by evaluating 3 concurrent token structures.
  * **`tokens`**: A list of exactly 3 active token IDs.
  * **`training_mode`**: If `True`, performs topological plasticity.
  * **`correct_target`**: The target index to wire active tokens to.
  * **`decay_factor`**: Background decay multiplier.
* **`decay(decay_factor: float) -> None`**
  * Shaves a percentage of integer value off the global accumulators.
* **`get_active_nodes() -> list[int]`**
  * Returns an array of currently active token IDs (accumulators > 0).
* **`get_loop_intensities() -> dict[int, int]`**
  * Returns a dictionary mapping active token IDs to their accumulator intensity.
* **`get_structural_mutations() -> list[tuple[int, int, int, int]]`**
  * Emits and drains structural mutation events.
  * **Returns**: A list of tuples: `(token_id, evicted_target, new_target, timestamp_us)`.

---

### 3. `PyInsectoidSimulation`
An advanced simulation of an insectoid rigid-body walking gait driven by the rhythmic resonance of feedback loops in the spatiotemporal mesh.

#### Constructor
* **`PyInsectoidSimulation()`**
  * *Instantiates a spatiotemporal mesh and configures rhythmic Central Pattern Generator (CPG) feedback loops.*

#### Methods
* **`step(training_mode: bool, correct_target: int | None = None, decay_factor: float = 0.9) -> None`**
  * Tokenizes the continuous insectoid state (phases and velocities) and steps the simulation.
* **`get_phases() -> list[float]`**
  * Returns the current phases of the 6 legs.
* **`get_velocities() -> list[float]`**
  * Returns the current velocities of the 6 legs.
* **`get_active_nodes() -> list[int]`**
  * Returns currently active token IDs in the underlying mesh.
* **`get_loop_intensities() -> dict[int, int]`**
  * Returns a map of loop intensities from the underlying mesh.
* **`get_structural_mutations() -> list[tuple[int, int, int, int]]`**
  * Returns structural mutation events from the underlying mesh.

---

## Performance & Benchmarks

`neuronguard` delivers orders-of-magnitude improvements in training speed, memory consumption, and model size compared to traditional deep learning libraries, while maintaining highly competitive accuracy.

### 1. Standard Dataset Benchmarks

| Dataset / Task | Samples | Classes | Sensory Neurons | Training Time | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AG News Dataset** | 120,000 | 4 | 1,000 | **1.27s** | **82.14%** |
| **DBpedia Ontology Dataset** | 560,000 | 14 | 5,000 | **0.64s** | **86.26%** |
| **Credit Card Fraud Scanner** | 29,222 | 2 | 50 | **0.025s (25ms)** | **99.93%** (75.00% Precision) |

*Benchmarks run on an Apple M2 Pro CPU using Python 3.12.*

### 2. Head-to-Head Comparison: NeuronGuard vs. PyTorch
This benchmark compares `neuronguard` against a standard PyTorch feedforward neural network trained on the **DBpedia Ontology Dataset** (560,000 training samples, 70,000 test samples, 14 classes).

| Metric | NeuronGuard | PyTorch (CPU) | Improvement |
| :--- | :--- | :--- | :--- |
| **Training Time** | **0.64s** | 5.30s | **8.28x Faster** |
| **Test Accuracy** | **86.26%** | 86.06% | **+0.20% Higher** |
| **Model Size (Disk)** | **313.4 KB** | 1253.6 KB | **4.0x Smaller** |
| **Memory Footprint** | **~320 KB** | ~500+ MB (Heap) | **~1500x Lower** |

*To reproduce this head-to-head comparison on your machine, run:*
```bash
mise run dbpedia_benchmark
```

---

## Core Concept & Architecture

At its core, `neuronguard` operates on a hardware-conscious, matrix-free neuromorphic design. Instead of performing heavy floating-point matrix multiplications (like traditional deep learning), it models intelligence as a network of **cache-aligned, thread-bounded neurons** that communicate via discrete event spikes.

### 1. Parallel Stream Ingestion & Routing
When a list of active sensory tokens is presented, the engine drops the Python GIL and broadcasts the active neurons to a parallel thread pool. Each thread evaluates a specific connection index in parallel, performing lock-free atomic additions directly onto the motor potentials.

```text
               [ Incoming Sensory Tokens ] (e.g., [42, 108, 512])
                           │
                           ▼
               ┌───────────────────────┐
               │  NeuronGuardField     │
               │ (Drops Python GIL)    │
               └───────────┬───────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   ┌───────────────────┐       ┌───────────────────┐
   │ Sensory Neuron 42 │       │ Sensory Neuron 108│  <─── Cache-Aligned (64-byte)
   │  (Cache Line 42)  │       │ (Cache Line 108)  │       ThreadBoundedNeurons
   └─────────┬─────────┘       └─────────┬─────────┘
             │                           │
             │ (Parallel Broadcast)      │ (Parallel Broadcast)
             ▼                           ▼
   ┌───────────────────────────────────────────────┐
   │                ParallelRouter                 │  <─── Drops GIL, lock-free
   │   [Thread 0]   [Thread 1]   ...   [Thread 7]  │       atomic evaluations
   └───────┬────────────┬─────────────────┬────────┘
           │            │                 │
           ▼            ▼                 ▼
   ┌───────────────────────────────────────────────┐
   │               Motor Potentials                │  <─── Atomic additions
   │  [Motor 0]     [Motor 1]    ...   [Motor N]   │       straight to memory
   └───────────────────────────────────────────────┘
                           │
                           ▼
              [ Winning Motor Neuron ID ] (Highest Potential)
```

### 2. Guard/Lease Transactional Plasticity
To perform safe concurrent learning (topological plasticity) without global locks, `neuronguard` uses a transactional lease pattern. Before mutating a neuron's connections, a thread must acquire a lock-free lease on that neuron's specific memory address using an atomic compare-and-swap (CAS) operation on its padding bytes.

```text
           ┌────────────────────────────────────────┐
           │ Attempt to Acquire Transactional Lease │
           └───────────────────┬────────────────────┘
                               │
                Atomic CAS on Padding Bytes (0 ➔ 1)
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
                 [Success]           [Failure]
                     │                   │
         ┌───────────┴───────────┐  ┌────┴────────────────────────┐
         │ Safe Concurrent       │  │ Skip or Retry               │
         │ Topological Plasticity│  │ (Prevents Write Contention) │
         └───────────┬───────────┘  └─────────────────────────────┘
                     │
          Release Lease (1 ➔ 0)
```

---

## Developer Tasks & Experiments

All experiments, benchmarks, and guides are located in the `example/` directory. You can list all available tasks by running `mise tasks` or simply run them cleanly using `mise run <task_name>`:

```bash
# List all available tasks and experiments
mise tasks

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

# Run the Wikipedia Structured Dataset Classifier & Router (Streaming 10.4M articles)
mise run wikipedia_classifier

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
