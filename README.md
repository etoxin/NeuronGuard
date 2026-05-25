# neuron-PoC

## Part 1: The PoC Structural Design

Your native Rust PoC should be designed around a **"Rhythm Tracker"**—a small network that takes a specific sequence of timed pulses (like `Dot-Dot-Dash`) and uses your `Guard` system to train the network to recognize it.

### 1. Memory Configuration (`src/memory.rs`)

This module enforces your strict 16-byte layout and pointerless offset arithmetic.

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

impl NeuronField {
    // Pure pointerless offset arithmetic mapping to Base + ID * 16
    pub unsafe fn get_neuron(&self, id: usize) -> &mut GuardedNeuron {
        &mut *self.storage.add(id)
    }
}

```

### 2. The Core Multi-Threaded Queue (`src/queue.rs`)

Manages your worker execution threads pulling from a thread-safe ring buffer (`crossbeam-channel`).

```rust
pub enum RuntimeMode {
    Run,
    Trainer,
}

pub struct EventPacket {
    pub target_id: u32,
    pub magnitude: f32,
    pub source_id: Option<u32>, // Only populated in Trainer Mode
}

```

### 3. The Guard Lifecycle Loop (`src/guard.rs`)

This is your transactional execution framework. When an event fires in Trainer Mode, it builds a localized execution scope.

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

## Part 2: The Successful PoC Checklist

A successful proof of concept means verifying your architectural constraints, not building a massive network. Treat this as your definition of "Done" tonight:

### Phase 1: The Bare-Metal Architecture Validation

* [ ] **Zero Global State Verification:** Your background worker threads are updating memory nodes completely via array index lookups (`NEURON_FIELD[id]`), without a single global read/write lock (`Mutex` or `RwLock`) wrapped around the array.
* [ ] **Compile-Time Size Enforcement:** A unit test using `std::mem::size_of::<GuardedNeuron>()` successfully confirms your memory layout is exactly 16 bytes.
* [ ] **Thread Independence:** Spin up 4 worker threads. Blast 10,000 independent event packets at random neuron IDs through the queue and verify that the Mac performance cores handle them simultaneously without a single thread panicking or colliding.

### Phase 2: The Dual-Mode Execution Validation

* [ ] **Run Mode Flight Test:** In `RuntimeMode::Run`, feed a spike cascade through a sequence of 5 nodes. Verify that the event packet payload contains zero origin trackers, and that the execution flies forward sequentially using nothing but lightning-fast index mutations.
* [ ] **Trainer Mode Guard Test:** In `RuntimeMode::Trainer`, trigger a cascade where Node 0 activates Node 1, which activates Node 2. Verify that Node 2 successfully passes a feedback signal backwards through the open session trace to update Node 0’s `weight` variable *before* the temporary thread lifecycle ends.

### Phase 3: The Learning Proof

* [ ] **The Convergence Win:** Feed a simple temporal pattern (e.g., a pulse at Time X and another at Time Y) into your network. Use your `Guard` feedback loop to verify that the target node's `weight` changes until it consistently filters out random noise and only triggers an output when the correct pattern hits it.

---

Fire up your terminal, build out the modules, and see how clean you can keep those raw memory blocks. Let me know when you run your first compilation check! Good luck tonight.
