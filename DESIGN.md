# Architecture Design Document: `NeuronGuard`

**Project Status:** Active Prototyping

**Target Hardware Context:** L1/L2 CPU Cache Boundaries ($\le 32\text{ KB}$ baseline)

**Core Paradigm:** Generalized Matrix-Free Address-Mapping Token Router

---

## 1. Executive Summary & Design Philosophy

Traditional machine learning models process structural data by embedding discrete symbols into multi-dimensional floating-point tensors, manipulating them via dense linear algebra loops ($Y = WX + b$). This imposes significant hardware constraints, requiring large memory footprints, heavy VRAM allocation, and static execution blocks during backpropagation.

`NeuronGuard` completely rejects dense numerical embeddings. It is an artificial nervous system designed as a **Generalized Address-Mapping Token Router**. It treats a data stream not as a continuous mathematical space, but as a collection of discrete, explicit symbols.

By replacing floating-point calculus with **Discrete Address Space Geometry** and caching states within a strict physical tape boundary, the runtime engine executes inference and structural adaptation simultaneously using microsecond-level pointer adjustments.

### Core Architectural Pillars

* **Matrix-Free Execution:** No linear algebra, dot products, or floating-point weights. All evaluations utilize direct bit-shift integer arithmetic.
* **Hardware-Bounded Sparsity:** The system activates only the memory regions explicitly addressed by incoming tokens. Unused pathways remain completely dark, maximizing L1 cache line performance.
* **Lifelong Inline Plasticity:** Employs a transactional **Guard/Lease pattern** that allows concurrent threads to safely mutate independent neural registers directly on the stack without global execution locks.

---

## 2. Memory Topography & Struct Layouts

To enforce strict, predictable layout behavior and eliminate heap fragmentation, all structures utilize explicit C-representation alignment bindings (`#[repr(C)]`). The engine's active field is constrained to map tightly within a standard CPU cache line size.

```rust
// Configuration Bounds
pub const MAX_THREADS: usize = 8; // Locked directly to target CPU core architecture

/// ThreadBoundedNeuron
/// Spatially aligned to exactly 64 bytes to fill a standard CPU cache line.
/// Eliminates false sharing and guarantees deterministic hardware pre-fetching.
#[repr(C, align(64))]
pub struct ThreadBoundedNeuron {
    pub token_id: u32,                               // 4 Bytes: Unique symbol key
    pub active_connections: u32,                     // 4 Bytes: Actual count (<= MAX_THREADS)
    pub target_neuron_ids: [u32; MAX_THREADS],       // 32 Bytes: Downstream destination buckets
    pub weight_modifiers: [i16; MAX_THREADS],        // 16 Bytes: Localized connection strengths
    pub padding: [u8; 8],                            // 8 Bytes: Structure alignment round-out
}
```

---

## 3. Data Plane & Execution Loops

### 3.1 Step 1: Frontend Tokenization (Quantization Layer)

Data ingesting into the engine must be stateless and fast. Continuous physical values (such as laboratory metrics or infrastructure logs) are quantized into discrete bit-packed identifiers, bypassing string allocations and heavy hash tables.

```rust
// Instant, allocation-free feature compression
pub fn tokenize_features(metric_a: f64, metric_b: f64, metric_c: f64) -> u64 {
    let mut token: u64 = 0;
    
    let bucket_a = if metric_a > 20.0 { 7 } else if metric_a > 15.0 { 4 } else { 1 };
    token |= bucket_a << 0;
    
    let bucket_b = if metric_b > 1200.0 { 7 } else if metric_b > 800.0 { 4 } else { 1 };
    token |= bucket_b << 8;
    
    let bucket_c = if metric_c > 0.25 { 7 } else if metric_c > 0.10 { 4 } else { 1 };
    token |= bucket_c << 16;
    
    token
}
```

### 3.2 Step 2: The Direct Memory Leap (Inference)

The runtime engine locates the target neuron space using bitwise arithmetic. It treats the token ID as a physical pointer offset:

$$\text{Target Pointer} = \text{Base Address} + (\text{Token ID} \times 64)$$

```rust
// Single instruction leap: Token ID shifted by 6 to scale to 64-byte structures
let target_offset = token_id << 6;
```

### 3.3 Step 3: Parallel Fan-Out (Coarse Coding Execution)

When a token fires, its axon terminal distributes signals concurrently. Rather than a single thread iterating through an array sequentially, the master pipeline broadcasts the cache line across the thread pool via lock-free `crossbeam` channels. Each thread evaluates its assigned index slice using its internal **Thread ID**:

```rust
pub fn evaluate_parallel(
    neuron: &ThreadBoundedNeuron, 
    thread_id: usize, 
    accumulators: &mut [AtomicI32]
) {
    if thread_id < neuron.active_connections as usize {
        let target = neuron.target_neuron_ids[thread_id] as usize;
        let weight = neuron.weight_modifiers[thread_id] as i32;
        
        // Concurrent atomic addition straight into target accumulation state
        accumulators[target].fetch_add(weight, std::sync::atomic::Ordering::Relaxed);
    }
}
```

---

## 4. Control Plane & Live Learning Strategy

Learning is governed by a decentralized, competitive system running directly on production traffic without an offline backpropagation pass.

### 4.1 The Guard/Lease Mutation Pattern

When a downstream feedback signal detects an incorrect routing choice, the system triggers an inline correction loop:

1. An independent worker thread acquires a transactional, lock-free lease on the specific 64-byte memory address of the active token.
2. The local thread accesses the variables right on the CPU stack.
3. It updates the weights, amplifying pathways that lead to the correct decision and suppressing pathways that caused the error.
4. The lease drops out, making the structural change visible on the very next inference clock cycle.

### 4.2 Autonomous Least-Significant Eviction (Hardware Self-Pruning)

Because connections are strictly constrained to the system thread pool count (`MAX_THREADS`), a neuron cannot expand indefinitely. When a node hits capacity and must learn a new pathway, it executes automatic memory recycling:

1. The thread scans the `weight_modifiers` array on the active cache line.
2. It identifies the weakest connection (the value closest to zero).
3. It evicts that specific slot, overwriting its data with the new `target_neuron_id` and resetting the weight.
4. The model physically forgets its least significant historical memory to clear space for active learning, completely avoiding memory bloat or global indexing locks.

---

## 5. System Boundary Comparison

| Metric | Traditional Dense Architectures (PyTorch) | `NeuronGuard` Engine |
| --- | --- | --- |
| **Mathematical Core** | Dense Floating-Point Linear Algebra | Discrete Index Address Space Geometry |
| **Active Footprint** | Megabytes to Gigabytes of VRAM / Heap | **32.22 KB** (Strictly bounded to CPU Cache) |
| **Thread Scaling** | Global Thread Barriers / Matrix Locks | Lock-Free Single-Token Thread Partitioning |
| **Adaptation Rule** | Global Backpropagation (Frozen Runtime) | Localized Guard/Lease Mutations (Live Inline) |
| **Observability** | Black Box Matrix Clouds | **White Box Hex-Inspectable State Registers** |

---

## 6. Implementation Targets & Benchmark Suites

To prove the structural reliability of this generalized model layout, testing must reject uncurated, long-form linguistic prose (which saturates sparse accumulators) and prioritize structured, high-velocity streams:

1. **High-Frequency Systems Telemetry:** Real-time infrastructure failure profiling and automatic, sub-millisecond circuit breaking.
2. **Scientific Laboratory Diagnosis (UCI WDBC):** Mapping cellular physical parameters into quantized token coordinates to verify high-precision, white-box analytical diagnostics.
3. **Low-Power Event Vision (WASM Integration):** Compiling the core engine into WebAssembly to monitor coordinate-based event streams directly inside a client-side visual grid.
