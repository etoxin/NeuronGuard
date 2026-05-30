# Implementation Tasks: NeuronGuard-Gen (v1.0-Alpha)

This document outlines the step-by-step engineering tasks required to implement the **NeuronGuard-Gen (v1.0-Alpha)** requirements specified in `PRD.md` into the existing `neuronguard` library.

---

## Architectural Decisions & Design Notes

### 1. Memory Layout & Cache-Line Alignment
The PRD specifies a `PermanentNeuromorphicLine` struct aligned to 64 bytes. However, let's analyze the size of the fields:
- `synapses_positive: [u32; 8]` (32 bytes)
- `synapses_negative: [u32; 8]` (32 bytes)
- `loopback_address: u32` (4 bytes)
- `loopback_energy: u8` (1 byte)
- `local_potential: i16` (2 bytes)
- `activation_threshold: i16` (2 bytes)
- `_padding: [u8; 18]` (18 bytes)

**Total size:** $32 + 32 + 4 + 1 + 2 + 2 + 18 = 91\text{ bytes}$.
Since $91\text{ bytes} > 64\text{ bytes}$, aligning this struct to 64 bytes will cause the Rust compiler to pad it to **128 bytes** (exactly two cache lines). 

**Proposed Solution:**
To maintain strict cache-line alignment and optimize memory footprint, we have two options:
1. **Option A (64-byte Single Cache Line):** Reduce the synaptic arrays to `[u32; 4]` (128 bits each, representing 128 positive and 128 negative pathways). This reduces the synapses to 32 bytes total, allowing the entire struct to fit in exactly 64 bytes with 27 bytes of padding.
2. **Option B (128-byte Dual Cache Line):** Keep `[u32; 8]` (256 bits each, representing 256 positive and 256 negative pathways) and adjust the padding to exactly 55 bytes (`_padding: [u8; 55]`) so the struct is exactly 128 bytes (two cache lines).

*Task 1.1 will implement Option B to preserve the 256-bit synaptic capacity specified in the PRD, while ensuring perfect dual cache-line alignment.*

---

## Phase 1: Low-Level Memory & Hardware Alignment (Rust)

### Task 1.1: Implement `PermanentNeuromorphicLine`
- **File:** `src/memory.rs` (or a new `src/gen_memory.rs` module)
- **Description:** Implement the `PermanentNeuromorphicLine` struct with strict `#[repr(align(64))]` alignment.
- **Details:**
  - Define the struct with `synapses_positive: [u32; 8]`, `synapses_negative: [u32; 8]`, `loopback_address: u32`, `loopback_energy: u8`, `local_potential: i16`, `activation_threshold: i16`, and `_padding: [u8; 55]` to make it exactly 128 bytes (perfectly aligning to two 64-byte cache lines).
  - Add a compile-time assertion in a unit test to verify that `std::mem::size_of::<PermanentNeuromorphicLine>() == 128` and `std::mem::align_of::<PermanentNeuromorphicLine>() == 64`.

### Task 1.2: Implement Lock-Free Atomic Potential State
- **File:** `src/guard.rs` or `src/memory.rs`
- **Description:** Implement the `AtomicPotentialState` struct to manage concurrent potential updates safely across threads without locks.
- **Details:**
  - Define `AtomicPotentialState` containing `pub potential: AtomicI16`.
  - Implement `try_lease_and_accumulate(&self, increment: i16)` using a Compare-And-Swap (`CAS`) loop with `compare_exchange_weak` and `Ordering::SeqCst` / `Ordering::Relaxed` as specified in the PRD.

---

## Phase 2: MatMul-Free Spiking Linear Attention (Rust)

### Task 2.1: Implement Ternary Synaptic Weight Masking
- **File:** `src/attention.rs` (New Module)
- **Description:** Implement the ternary synaptic evaluation logic ($W \in \{-1, 0, 1\}$) using binary masks.
- **Details:**
  - Implement a function `evaluate_ternary_synapses(line: &PermanentNeuromorphicLine, input_spikes: &[u32; 8]) -> i16` that:
    - Performs bitwise AND between `input_spikes` and `line.synapses_positive` to find active excitatory pathways (each active bit adds `+1` to the accumulator).
    - Performs bitwise AND between `input_spikes` and `line.synapses_negative` to find active inhibitory pathways (each active bit subtracts `-1` from the accumulator / inverts sign).
    - Bypasses floating-point ALUs entirely by using population count (`.count_ones()`) on the resulting bitwise masks.

### Task 2.2: Implement Spiking Linear Attention Stream
- **File:** `src/attention.rs`
- **Description:** Implement the recurrent linear attention stream: $\text{Attention}(Q, K, V) = Q \times (K^T \times V)$.
- **Details:**
  - Implement the attention pooling calculation ($K^T \times V$) as a sparse array of integer additions.
  - Since weights are ternary, replace the quadratic matrix multiplication with sparse pointer leaps and additions.
  - Ensure zero heap allocations in the hot path by using pre-allocated arrays or stack-resident buffers.

---

## Phase 3: Central Pattern Generators & Leak Dynamics (Rust)

### Task 3.1: Implement Recurrent Echoes (CPG Loopback)
- **File:** `src/cpg.rs` (New Module)
- **Description:** Implement working memory across paragraph boundaries using dedicated backward-routing slots inside every active neuromorphic line.
- **Details:**
  - When a localized domain pattern spikes, it echoes energy backward to preceding cache locations using `loopback_address` and `loopback_energy`.
  - Implement a function `propagate_cpg_echo(line: &mut PermanentNeuromorphicLine, memory_pool: &mut [PermanentNeuromorphicLine])` that transfers `loopback_energy` to the target `loopback_address`.

### Task 3.2: Implement Continuous Synaptic Leak Loop
- **File:** `src/cpg.rs`
- **Description:** Implement an atomic background loop that applies a fixed decay factor ($\alpha$) on every clock cycle pass to prevent accumulator saturation.
- **Details:**
  - Implement `decay_potentials(lines: &mut [PermanentNeuromorphicLine], alpha: f32)` to decay the `local_potential` and `loopback_energy` of all lines.
  - Ensure the decay operation is performed using atomic operations or safe lock-free mutations.

---

## Phase 4: Hierarchical Winner-Take-All Selector (Rust)

### Task 4.1: Implement Tier-1 Macro Sieve
- **File:** `src/wta.rs` (New Module)
- **Description:** Implement the first tier of the cascading search to isolate the target context down to a highly constrained grammatical cluster.
- **Details:**
  - Group the 50,000 target tokens into macro clusters (e.g., grammatical or semantic categories).
  - Implement a fast bitmask-based or sparse-addition-based sieve to identify the active macro cluster with the highest aggregate potential.

### Task 4.2: Implement Tier-2 Micro Target Selector
- **File:** `src/wta.rs`
- **Description:** Implement the second tier of the cascading search to activate a hyper-specific, 64-byte aligned memory block containing the final target token indices.
- **Details:**
  - Once the macro cluster is isolated, search only the corresponding micro target block.
  - Return the final target token indices to pass back up through the FFI layer.

---

## Phase 5: Subword Tokenization Frontend (Python)

### Task 5.1: Implement Highly Compressed BPE Dictionary
- **File:** `python/neuronguard/tokenizer.py` (New File)
- **Description:** Create a local Byte-Pair Encoding (BPE) tokenizer mapping up to 50,000 subword fragments down to flat `u16` Token IDs.
- **Details:**
  - Implement a lightweight BPE vocabulary loader and encoder/decoder.
  - Ensure the vocabulary is saved in a highly compressed format (e.g., JSON or binary).

### Task 5.2: Implement Semantic Field Splitter
- **File:** `python/neuronguard/tokenizer.py`
- **Description:** Implement the concurrent split of input text across three semantic fields.
- **Details:**
  - **Field 0 (Lexical):** Maps word fragments directly to Token IDs.
  - **Field 1 (Formatting/Density):** Quantizes structural features like punctuation density, capitalizations, and indentation.
  - **Field 2 (Micro-temporal Distance):** Computes distance profiles (token counts) between adjacent nouns/entities.
  - Return a tuple of lists or a structured array containing the three fields.

---

## Phase 6: Conversational State Execution & Sampling Loop (Python & FFI)

### Task 6.1: Expose isolated execution context to Python
- **File:** `src/lib.rs`
- **Description:** Update the PyO3 bindings to support isolated, session-specific context runners.
- **Details:**
  - Ensure that the base weights of the model remain read-only.
  - Expose a method to initialize a session-specific runner containing its own current accumulator potentials and memory states.

### Task 6.2: Implement Autoregressive Generation Loop
- **File:** `python/neuronguard/runner.py` (New File)
- **Description:** Implement the autoregressive generation loop with stochastic temperature scaling and Top-10 sampling.
- **Details:**
  - Implement `run_autoregressive_generation(prompt_token_ids, max_generation_length=100, temperature=0.7)` as specified in the PRD.
  - Step a single clock cycle pass, pull potentials, apply temperature, filter with Top-10 sampling, and sample the next token ID.

---

## Phase 7: Verification & Testing

### Task 7.1: Rust Unit Tests
- Add unit tests in `src/attention.rs`, `src/cpg.rs`, and `src/wta.rs` to verify:
  - Perfect cache-line size and alignment of all structures.
  - Correctness of the ternary synaptic masking and sparse additions.
  - Correctness of the CPG echo propagation and decay dynamics.
  - Correctness of the two-tier WTA selector.

### Task 7.2: Python Integration Tests
- Create an integration test script `tests/test_generation.py` to verify:
  - Tokenizer correctly splits text into the three semantic fields.
  - Autoregressive generation loop runs successfully and outputs coherent token sequences.
  - Isolated session runners do not leak state or corrupt base weights.

### Task 7.3: Performance Benchmarking
- Measure and verify the targeted performance milestones:
  - Zero heap allocations in the inference path.
  - Sub-millisecond Time-To-First-Token (TTFT) generation.
  - Total memory footprint of the active execution context.
