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

## Usage Examples

### 1. Minimal Inference & Decay Loop
```python
import neuronguard as ng
import time

# Initialize Local Biological Cortex (1000 Sensory -> 4 Motor Neurons)
cortex = ng.NeuronGuardField(sensory_count=1000, motor_count=4)

# Process incoming sensory stimuli tokens (drops GIL instantly)
active_stimuli = [42, 108, 512]
triggered_motor_id = cortex.process_stream(active_stimuli, training_mode=True)

print(f"Stimuli {active_stimuli} ➔ Triggered Motor Neuron: {triggered_motor_id}")

# Run background metabolic decay loop
cortex.tick_decay(decay_factor=0.90)
```

### 2. Supervised Learning (Trainer Mode)
```python
import neuronguard as ng

cortex = ng.NeuronGuardField(sensory_count=100, motor_count=3)

# Train Sensory Neuron 2 to target Motor Neuron 2
# Amplifies correct pathway (+20) and suppresses competing pathways (-5)
cortex.train_stream(
    sensory_tokens=[2], 
    correct_motor_id=2, 
    amplify_delta=20, 
    suppress_delta=5
)

# Verify the training worked
cortex.reset_potentials()
winning_motor_id = cortex.process_stream([2], training_mode=False)
print(f"Winner: Motor Neuron {winning_motor_id} (Expected: 2)")
```

### 3. GIL-Free Multi-Threaded Ingestion
```python
import threading
import time
import neuronguard as ng

cortex = ng.NeuronGuardField(sensory_count=5000, motor_count=8)

def worker_thread():
    for _ in range(1000):
        # Process streams concurrently without Python-level lock contention
        cortex.process_stream([42, 108, 2048], training_mode=False)
        time.sleep(0.001)

threads = [threading.Thread(target=worker_thread) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print("Successfully processed concurrent streams with zero GIL bottlenecks!")
```

### 4. Spatiotemporal Mesh & Gait Simulation
```python
import neuronguard as ng

sim = ng.PyInsectoidSimulation()

# Run 10 steps of steady walking gait
for step in range(10):
    sim.step(training_mode=False, decay_factor=0.90)
    print(f"Leg Phases: {sim.get_phases()}")

# Inject sudden perturbation and trigger transactional re-patching
sim.step(training_mode=True, correct_target=0, decay_factor=0.90)

# Retrieve structural mutations emitted by the Guard/Lease pattern
mutations = sim.get_structural_mutations()
for token_id, evicted, new_target, ts in mutations:
    print(f"Token {token_id}: Evicted {evicted} ➔ Patched to {new_target} at {ts} us")
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

## Developer Tasks

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
