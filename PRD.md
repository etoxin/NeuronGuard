### Product Design Document: NeuronGuard-Gen (v1.0-Alpha)

**Project Title:** MatMul-Free Spiking Large Language Model Core

**Author:** Lead Software Engineer

**Status:** Draft / Active Specification

**Implementation Language:** Rust (with zero-overhead Python FFI bindings via `pyo3`)

---

## 1. Executive Summary & Design Constraints

NeuronGuard-Gen is a hardware-conscious, autoregressive, generative Spiking Neural Network Language Model (SNN-LM). It scales the original NeuronGuard event-driven classification paradigm into a low-latency, conversational chat intelligence.

The core architectural mission remains absolute: **Compute must adapt to the physical constraints of localized silicon.** The system replaces continuous floating-point Matrix Multiplications ($\text{MatMul}$) with discrete binary event spikes, ternary synaptic weights ($\{-1, 0, 1\}$), and strict 64-byte hardware cache alignment.

### Core Hard Constraints:

1. **Zero Heap Allocations in the Inference Path:** All memory states must be pre-allocated and stack- or static-resident.
2. **Cache-Line Bound:** Every neural execution line must map to exactly 64 bytes to prevent cache thrashing and maximize L1/L2 data locality on standard CPU performance cores.
3. **No External Deep Learning Dependencies:** Zero reliance on PyTorch, LibTorch, or tensor runtimes. Compiled completely via raw Rust primitives.

---

## 2. System Architecture & High-Level Data Flow

The system transitions from an entire-document evidence sieve to a continuous, sequential autoregressive clock cycle operating at linear $O(N)$ computational complexity.

```
      [ Input Context String / Prompt ]
                     │
                     ▼
  [ Python Frontend: BPE/N-Gram Tokenizer Sieve ] ➔ Chunks text to 16-bit Token IDs
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ RUST NEURONGUARD CORE ENGINE (L1/L2 Cache Resident Runtime)                  │
│                                                                              │
│  1. Spiking Linear Attention Layer (MatMul-Free Pipeline)                    │
│     Evaluates incoming token spikes sequentially: S_t = S_{t-1} + (K_t^T × V_t)│
│     * Operations map entirely to integer pointer leaps and sparse additions. │
│                                                                              │
│  2. Central Pattern Generators (Recurrent State Echo Loops)                  │
│     Propagates dynamic context backward to maintain short-term memory.       │
│                                                                              │
│  3. Continuous Synaptic Leak / Decay Loop                                    │
│     Shaves energy trail on every single execution pass: E_t = E_{t-1} × α    │
└──────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
 [ Hierarchical Winner-Take-All Selector ] ➔ Two-tier cascading tree search
                     │
                     ▼
      [ Raw Potential Index Array Out ]
                     │
                     ▼
  [ Python Runner Layer: Stochastic Sampling ] ➔ Temperature/Top-P Filter Block
                     │
                     ▼
       [ Emitted Next Token ID ] ──( Appends into Sensory Array Input Loop )──

```

---

## 3. Core Component Specifications

### 3.1 Subword Tokenization Frontend (Python Pre-Processor)

To completely decouple structural linguistic text parsing from the raw execution core, vocabulary tokenization is managed at the input script layer.

* **Mechanism:** A highly compressed, local Byte-Pair Encoding (BPE) dictionary mapping up to 50,000 subword fragments down to flat `u16` Token IDs.
* **Topological Fields:** Input text is split concurrently across three semantic fields:
* *Field 0:* Lexical word fragments.
* *Field 1:* Formatting syntax and structure density.
* *Field 2:* Micro-temporal distance profiles between adjacent nouns/entities.



### 3.2 MatMul-Free Spiking Linear Attention (Rust Layer)

Instead of matching every token against every other token over a quadratic $O(N^2)$ grid, attention is modeled as a recurrent linear stream:

$$\text{Attention}(Q, K, V) = Q \times (K^T \times V)$$

* **Ternary Synaptic Weights ($W \in \{-1, 0, 1\}$):** Multi-bit floating-point weights are completely eliminated. Sensory state evaluations are binary masked:
* `1`: Add sensory activation energy directly to the accumulator.
* `-1`: Invert sign bit / subtract energy (inhibitory loop).
* `0`: Skip memory address evaluation entirely.


* **Integer Additions Core:** The attention pooling calculation ($K^T \times V$) operates as a sparse array of integer additions, entirely skipping the floating-point ALUs of the processor.

### 3.3 Central Pattern Generators (CPGs) & Leak Dynamics

* **Recurrent Echoes:** Working memory across paragraph boundaries is maintained by dedicating backward-routing slots inside every active neuromorphic line. When a localized domain pattern spikes, it echoes energy backward to preceding cache locations.
* **Leak Channels:** To prevent the 256 global memory accumulators from saturating, an atomic background loop applies a fixed decay factor ($\alpha$) on every clock cycle pass, organically dissolving past semantic context trails as new information streams in.

### 3.4 Hierarchical Winner-Take-All (WTA) Tree Selector

Searching a global pool of 50,000 target tokens in a single pass would shatter L1/L2 cache boundaries. NeuronGuard-Gen implements a two-tier cascading search:

1. **Tier-1 (Macro Sieve):** Spikes isolate the target context down to a highly constrained grammatical cluster (e.g., *Nouns related to Science*, *Core Verbs*, *Structural Punctuation*).
2. **Tier-2 (Micro Target):** Activates a hyper-specific, 64-byte aligned memory block containing the final target token indices to pass back up through the FFI layer.

---

## 4. Memory Layout & Low-Level Data Structures

The Rust memory engine enforces cache-line isolation. No pointers are allowed inside the main execution lines; all state connections are tracked as relative array offsets.

```rust
/// Enforce strict 64-byte alignment to match standard CPU cache lines.
/// This maximizes L1/L2 data locality and prevents cache line thrashing.
#[repr(align(64))]
pub struct PermanentNeuromorphicLine {
    /// 256 bits representing active excitatory synaptic pathways
    pub synapses_positive: [u32; 8], 
    
    /// 256 bits representing active inhibitory synaptic pathways
    pub synapses_negative: [u32; 8], 
    
    /// Relative offset pointer for Central Pattern Generator backward routing
    pub loopback_address: u32,
    
    /// Remaining lingering energy amplitude inside the recurrent loop
    pub loopback_energy: u8,
    
    /// Current integer accumulation potential
    pub local_potential: i16,
    
    /// Dynamic activation threshold before a spike event is triggered
    pub activation_threshold: i16,
    
    /// Strict padding to guarantee that instances align perfectly to hardware bounds
    pub _padding: [u8; 18],
}

```

### Concurrent Weight Mutation Safety:

To manage concurrent updates safely across background threads without introducing locks, thread synchronization uses the atomic **Guard/Lease** transactional pattern. Weight changes are checked at the hardware instruction level using atomic Compare-And-Swap (`CAS`) loops on individual memory registers:

```rust
use std::sync::atomic::{AtomicI16, Ordering};

pub struct AtomicPotentialState {
    pub potential: AtomicI16,
}

impl AtomicPotentialState {
    pub fn try_lease_and_accumulate(&self, increment: i16) {
        let mut current = self.potential.load(Ordering::Relaxed);
        loop {
            let target = current.saturating_add(increment);
            match self.potential.compare_exchange_weak(
                current,
                target,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

```

---

## 5. Conversational State Execution & Sampling Loop

To ensure multiple concurrent chat interactions can happen without corrupting the model's primary weights, the base weights remain read-only. Every chat session passes an isolated, stack-allocated context runner containing its own current accumulator potentials and memory states.

```python
# python/neuronguard/runner.py
import numpy as np

def run_autoregressive_generation(prompt_token_ids, max_generation_length=100, temperature=0.7):
    # Initialize an isolated execution context inside the Rust layer
    session_field = ng.NeuronGuardField(vocab_size=50000, motor_count=50000)
    session_field.reset_potentials()
    
    # Process the seed prompt to establish initial CPG loopback energy
    session_field.process_stream_sync(prompt_token_ids)
    
    generated_sequence = []
    current_token_id = prompt_token_ids[-1]
    
    for _ in range(max_generation_length):
        # Step a single clock cycle pass
        session_field.process_stream_sync([current_token_id])
        
        # Pull raw integer potential balances directly via pointer view
        raw_potentials = np.array(session_field.get_potentials(), dtype=np.float32)
        
        # Apply stochastic Temperature layer to introduce lexical variance
        scaled_logits = raw_potentials / max(temperature, 1e-5)
        probabilities = exp_softmax(scaled_logits)
        
        # Top-10 sampling pool filter to isolate coherent next-tokens
        top_indices = np.argpartition(probabilities, -10)[-10:]
        top_probs = probabilities[top_indices]
        top_probs /= top_probs.sum()
        
        sampled_token_id = np.random.choice(top_indices, p=top_probs)
        
        if sampled_token_id == END_OF_SEQUENCE_MARKER:
            break
            
        generated_sequence.append(sampled_token_id)
        current_token_id = sampled_token_id # Seed back into sensory array
        
    return generated_sequence

```

---

## 6. Targeted Performance Milestones

By optimizing data structures directly for standard instruction architectures, the 1.5-Billion parameter execution target operates inside an unprecedented resource envelope:

* **Compute Type:** 100% MatMul-Free Sparse Integer Additions.
* **Inference Pipeline Profile:** Fully linear $O(N)$ uniform clock cycle pass.
* **Operational Latency:** Sub-millisecond Time-To-First-Token (TTFT) generation.
* **Hardware Profile:** Fully functional inside standard x86_64 or Apple Silicon CPU caches (L2/L3) at a targeted structural power draw of **< 15 Watts**.