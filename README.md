# NeuronGuard

A high-performance, native Rust spiking neural network (SNN) designed for lock-free, asynchronous event routing and real-time local learning. By rejecting the traditional, GPU-heavy dense tensor paradigm, NeuronGuard demonstrates how neural processing and context routing can be executed at raw hardware limits on standard CPUs.

---

## Summary

**NeuronGuard** is built around a flat, 16-byte aligned memory field and a transactional, stack-allocated **Lease (Guard) Pattern** to achieve ultra-fast local learning and inference with zero global locks.

### Key Architectural Achievements:
* **Zero Global State Concurrency**: Background worker threads update memory nodes completely via array index lookups without a single global read/write lock (`Mutex` or `RwLock`), achieving true multi-core parallelism.
* **Cache-Optimal Memory Layout**: Neurons are represented as flat, 16-byte aligned blocks. Pointerless offset arithmetic ($Base + ID \times 16$) ensures perfect CPU L1/L2 cache-locality and zero pointer-chasing latency.
* **Transactional Stack-Allocated Leases**: Real-time learning is executed via a unique **Lease (Guard) Pattern** allocated on the thread's stack. Feedback signals propagate backwards along active pathways, and Rust's `Drop` trait automatically resets potentials, priming memory blocks with zero garbage collection overhead.
* **Production-Grade Scalability**: Proven beyond a basic PoC by training on the full **560,000-sample DBpedia Ontology dataset in under 20 seconds** on an Apple M2 Pro CPU, achieving **83.10% accuracy** with a compiled model size of only **32KB** (serializable and loadable in `< 1ms`).

---

## Performance Benchmarks

| Dataset / Task | Samples | Classes | Neuron Field Size | Training Time | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rhythm Tracker (Core PoC)** | - | 1 | 3 neurons | < 0.001s | **100%** (Converged) |
| **AG News Dataset** | 120,000 | 4 | 1,004 neurons | **3.38s** | **80.17%** |
| **DBpedia Ontology Dataset** | 560,000 | 14 | 2,014 neurons | **19.90s** | **83.10%** |

*Benchmarks run on an Apple M2 Pro CPU.*

```
MEMORY CONSUMPTION

NeuronGuard █ 32.22 KB
            
PyTorch     ██████████████████████████████████████████████████ est 500+ MB (Minimum runtime heap)
```

---

## How to Run

This project uses **`mise`** to manage toolchains and tasks.

### 0. Download the Datasets
Before running the AG News or DBpedia classifiers, download and extract the datasets:
```bash
mise run download_data
```

### 1. Run the Core PoC (Rhythm Tracker)
To run the pristine core PoC demonstrating temporal pattern convergence:
```bash
mise run poc
```

### 2. Run the Mixture-of-Experts (MoE) Router Example
To run the 105-neuron MoE router simulation:
```bash
mise run llm_router
```

### 3. Run the AG News 120,000 Sample Classifier
To download the dataset, train on 120,000 samples, and evaluate on 7,600 test samples:
```bash
mise run ag_news
```

### 4. Run the DBpedia 560,000 Sample Classifier
To download the dataset, train on 560,000 samples in under 20 seconds, and evaluate on 70,000 test samples:
```bash
mise run dbpedia
```

### 5. Run the Entire Test Suite
To run all 21 unit and integration tests across all binaries:
```bash
mise run test
```

---

## Architectural Overview

### 1. Memory Configuration (`src/memory.rs`)
Enforces a strict 16-byte layout and pointerless offset arithmetic.
```rust
// Force alignment to 16 bytes in memory
#[repr(C, align(16))]
pub struct GuardedNeuron {
    pub potential: f32,       // 4 bytes
    pub threshold: f32,       // 4 bytes
    pub target_id: u32,       // 4 bytes
    pub weight: f32,          // 4 bytes
} // Total = 16 bytes

pub struct NeuronField {
    pub storage: *mut GuardedNeuron,
    pub size: usize,
}
```

### 2. The Core Multi-Threaded Queue (`src/queue.rs`)
Manages worker execution threads pulling from a thread-safe lock-free ring buffer (`crossbeam-channel`).
```rust
pub enum RuntimeMode {
    Run,
    Trainer,
}

pub struct EventPacket {
    pub target_id: u32,
    pub magnitude: f32,
    pub source_id: Option<u32>,
}
```

### 3. The Guard Lifecycle Loop (`src/guard.rs`)
A transactional execution framework. When an event fires in Trainer Mode, it builds a localized execution scope on the stack.
```text
                  [ Incoming Event Packet ]
                             │
                             ▼
         [ Worker Thread Allocates Temporary Guard Scope ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ Apply Potential ]               [ Evaluate Threshold ]
  Modifies local 16 bytes            If Fired, extend Guard Chain
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
    [ Outcome Evaluated / Propagate Guard Backwards ]
    Traverses open session path -> Commits weight adjust
                             │
                             ▼
             [ Guard Automatically Drops ]
           Primes memory block for next thread
```

---

## License

This project is licensed under the **Apache License 2.0**. 

Under this license, you are free to copy, modify, distribute, and sell this software, including for commercial, closed-source products, provided that you include the original copyright and license notice. See the [LICENSE](LICENSE) file for the full license text.
