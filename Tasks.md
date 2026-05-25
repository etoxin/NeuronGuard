# PoC Implementation Plan & Tasks

Welcome to Rust! Since you come from a TypeScript background, this plan is designed to map Rust's low-level systems concepts (like raw pointers, threads, and memory alignment) to concepts you already know (like `ArrayBuffer`, worker threads, and async event queues).

We will build this Proof of Concept step-by-step. Each task is broken down with clear explanations, TypeScript analogies, and exact files to modify.

---

## 📋 The Master Checklist

### 🛠️ Setup & Infrastructure
- [x] **Task 0: Initialize Cargo Project**
  - Create `Cargo.toml` with necessary dependencies (`crossbeam-channel`, `rand`, etc.).

### 🧠 Phase 1: Bare-Metal Memory & Queue
- [x] **Task 1: Implement `src/memory.rs` (The Flat Memory Field)**
  - Define the 16-byte aligned `GuardedNeuron` struct.
  - Implement `NeuronField` using raw pointers and offset arithmetic.
  - **TS Analogy:** Think of this as a `Float32Array` backed by a single shared `SharedArrayBuffer`, where we read/write at exact byte offsets.
- [x] **Task 2: Implement `src/queue.rs` (The Lock-Free Event Queue)**
  - Define `EventPacket` and `RuntimeMode`.
  - Set up the multi-threaded worker pool using lock-free channels (`crossbeam-channel`).
  - **TS Analogy:** Think of this as a Node.js `Worker` pool pulling tasks from a thread-safe message channel.
- [x] **Task 3: Phase 1 Validation (Tests)**
  - Write a compile-time size check test (verifying `GuardedNeuron` is exactly 16 bytes).
  - Write a multi-threaded stress test (4 threads, 10,000 events) to verify zero-lock thread independence.

### 🛡️ Phase 2: Dual-Mode Execution & The Guard Primitive
- [x] **Task 4: Implement `src/guard.rs` (The Transactional Guard)**
  - Create the `Guard` struct that represents an active execution scope.
  - Implement the forward propagation and backward feedback trace.
  - **TS Analogy:** Think of a `Guard` as a transaction context (like a database transaction or a nested promise chain) that tracks the path of a request so it can roll back or commit changes to the source nodes.
- [x] **Task 5: Implement Run Mode & Trainer Mode in `src/main.rs`**
  - Implement the fast-path execution for `RuntimeMode::Run` (zero tracking overhead).
  - Implement the guarded execution for `RuntimeMode::Trainer` (tracking active pathways).
- [x] **Task 6: Phase 2 Validation (Tests)**
  - Write a Run Mode flight test (5-node cascade, zero origin trackers).
  - Write a Trainer Mode guard test (Node 0 -> Node 1 -> Node 2, verifying backward feedback updates Node 0's weight).

### 🎯 Phase 3: The Learning Proof (Rhythm Tracker)
- [x] **Task 7: Implement the Temporal Rhythm Tracker**
  - Feed a simple temporal pattern (e.g., pulses at specific intervals) into the network.
  - Train the network using the `Guard` feedback loop to filter out random noise and only fire on the correct rhythm.
- [x] **Task 8: Phase 3 Validation (The Convergence Win)**
  - Verify that the target node's weight successfully converges to recognize the target pattern.

---

## 🔍 Detailed Task Breakdown

### Task 0: Initialize Cargo Project
Rust uses **Cargo** for dependency management and building (similar to `npm` and `package.json`).
* **File to create:** `Cargo.toml`
* **Dependencies needed:**
  * `crossbeam-channel` (for lock-free multi-producer multi-consumer queues).
  * `rand` (for generating test patterns and noise).

---

### Task 1: Implement `src/memory.rs`
In TypeScript, objects are managed by a garbage collector and can be located anywhere in memory. In Rust, we can lay out structures *exactly* how we want them in physical RAM.

* **Concept:** We want a 16-byte struct aligned to 16 bytes.
* **TypeScript Analogy:**
  ```typescript
  // If we did this in JS/TS with an ArrayBuffer:
  const buffer = new ArrayBuffer(16 * totalNeurons);
  const view = new DataView(buffer);
  // Neuron ID 5 potential is at byte offset: 5 * 16 + 0
  // Neuron ID 5 threshold is at byte offset: 5 * 16 + 4
  ```
* **Rust Implementation Plan:**
  * Use `#[repr(C, align(16))]` to force the compiler to align the struct to 16 bytes.
  * Implement `NeuronField` with a raw pointer `*mut GuardedNeuron`.
  * Implement `unsafe fn get_neuron(&self, id: usize) -> &mut GuardedNeuron` using pointer arithmetic (`self.storage.add(id)`).

---

### Task 2: Implement `src/queue.rs`
Instead of using a single-threaded event loop like Node.js, Rust allows true multi-core parallelism. To prevent threads from fighting over data, we use a lock-free channel.

* **Concept:** A thread-safe ring buffer where input events are pushed, and worker threads pop them to process them.
* **TypeScript Analogy:**
  ```typescript
  // Like posting messages to a pool of Web Workers:
  worker.postMessage({ target_id: 5, magnitude: 1.5 });
  ```
* **Rust Implementation Plan:**
  * Define `EventPacket` with `target_id`, `magnitude`, and `source_id` (optional).
  * Set up a worker pool function that spawns $N$ threads. Each thread loops, waiting for events from a `crossbeam_channel::Receiver`, and processes them.

---

### Task 4: Implement `src/guard.rs`
The **Guard** is the heart of our learning mechanism. When an event propagates through the network, we need to remember the path it took so we can send a feedback signal backwards to adjust weights.

* **Concept:** A transactional scope. As a signal moves `Node 0 -> Node 1 -> Node 2`, we build a linked chain of "Guards". When the end is reached, we traverse this chain backwards to update weights, then drop the guards to release memory.
* **TypeScript Analogy:**
  Think of this like a call stack or a middleware chain where each step holds a reference to the previous step:
  ```typescript
  class Guard {
    constructor(
      public neuronId: number,
      public parent: Guard | null
    ) {}
  }
  ```
* **Rust Implementation Plan:**
  * Create a `Guard` struct that holds a reference/pointer to the neuron being modified and an optional reference to the parent `Guard` (the source of the activation).
  * Implement a backward propagation method on the guard chain that adjusts the `weight` of the source neurons based on whether the final activation was successful.

---

### Task 7: Implement the Rhythm Tracker
To prove the network can actually learn, we will build a **Rhythm Tracker**.
* **The Goal:** Recognize a `Dot-Dot-Dash` rhythm (e.g., pulses at intervals of 100ms, 100ms, 300ms).
* **How it learns:**
  * If the network fires when the rhythm is *incorrect* (noise), the Guard feedback loop decreases the weights of the active pathways.
  * If the network fires when the rhythm is *correct*, the Guard feedback loop increases the weights.
  * Over time, the weights converge so that only the correct temporal pattern triggers the final neuron.
