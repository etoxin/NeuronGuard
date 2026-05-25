# LLM-Guarded Event Engine PoC

A high-performance, native Rust Proof of Concept (PoC) for an **LLM-guarded event engine**. This project treats neural processing and context routing as a **systems programming and routing problem** rather than a massive global matrix transformation, utilizing a flat 16-byte aligned memory field and a transactional stack-allocated **Lease (Guard) Pattern** for lock-free, ultra-fast local learning.

---

## 📋 The Successful PoC Checklist (Completed!)

### Phase 1: The Bare-Metal Architecture Validation
* [x] **Zero Global State Verification:** Background worker threads update memory nodes completely via array index lookups (`NEURON_FIELD[id]`), without a single global read/write lock (`Mutex` or `RwLock`) wrapped around the array.
* [x] **Compile-Time Size Enforcement:** A unit test using `std::mem::size_of::<GuardedNeuron>()` successfully confirms the memory layout is exactly 16 bytes.
* [x] **Thread Independence:** Spun up 4 worker threads, blasted 10,000 independent event packets at random neuron IDs through the queue, and verified simultaneous lock-free processing without a single panic or collision.

### Phase 2: The Dual-Mode Execution Validation
* [x] **Run Mode Flight Test:** In `RuntimeMode::Run`, fed a spike cascade through a sequence of 5 nodes. Verified that the event packet payload contains zero origin trackers, and that execution flies forward sequentially using lightning-fast index mutations.
* [x] **Trainer Mode Guard Test:** In `RuntimeMode::Trainer`, triggered a cascade where Node 0 activates Node 1, which activates Node 2. Verified that Node 2 successfully passes a feedback signal backwards through the open session trace to update Node 0’s `weight` variable before the temporary thread lifecycle ends.

### Phase 3: The Learning Proof
* [x] **The Convergence Win:** Fed a simple temporal pattern into the network. Used the `Guard` feedback loop to verify that the target node's weight successfully converges to filter out random noise and only trigger an output when the correct pattern hits it.

---

## ⚡ Performance Benchmarks

| Dataset / Task | Samples | Classes | Neuron Field Size | Training Time | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rhythm Tracker (Core PoC)** | - | 1 | 3 neurons | < 0.001s | **100%** (Converged) |
| **AG News Dataset** | 120,000 | 4 | 1,004 neurons | **3.38s** | **80.17%** |
| **DBpedia Ontology Dataset** | 560,000 | 14 | 2,014 neurons | **19.90s** | **83.10%** |

*Benchmarks run on an Apple M2 Pro CPU.*

---

## 🚀 How to Run

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

## 🧠 Architectural Overview

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
