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

## Tasks

You can run all library and extension tasks cleanly using `mise`:

```bash
# Run the Getting Started Guide
mise run getting_started

# Run the Advanced Multi-Threaded Simulation
mise run advanced

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
