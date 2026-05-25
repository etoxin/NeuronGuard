# The Guard Primitive: Transactional Session Lifecycles

In fully decentralized networks, the hardest problem to solve is local credit assignment (how a neuron early in a chain discovers it contributed to a success or failure deep down the line). Traditional AI uses global backpropagation, which requires a complete stop and a massive reverse calculus pass over the whole model.

This framework introduces the **Guard Pattern**—a systems-engineering solution that models neural pathways like stateful, temporary telecommunication circuits.

## The Dual-Mode Runtime Lifecycle

To maximize efficiency, the engine splits execution into two entirely distinct modes depending on the system's current goal:

### 1. Trainer Mode (Guarded Borrowing)
When learning, every event is instantiated as an exclusive resource **Guard** (or Lease). 
* As a signal propagates forward, the thread creates a stateful execution scope that holds a temporary return path open.
* The 16-byte neuron blocks along the active pathway are marked as borrowed and locked.
* When the pathway hits an endpoint or evaluates an outcome, a feedback signal travels backwards up the open transactional trace, adjusting local `weight` variables.
* Once the transaction resolves, the guard scope drops out of execution, and the pristine memory blocks are placed safely back into the available pool.


```

[Event Triggered] ──> [Instantiate Guard Scope] ──> [Lock & Borrow 16-Bytes]
│
[Guard Scope Automatically Drops] <── [Commit Weight Adjust] <── [Process Return Pass]

```

### 2. Run Mode (Pure Flying)
Once pathways are sculpted, the engine strips the scaffolding away. 
* All leasing, tracking, and source payloads are completely deactivated.
* Event packets shrink to their absolute bare-minimum size: `(Target_ID, Magnitude)`.
* Worker threads perform raw, unidirectional index mutations. The signal literally "flies" through memory at absolute hardware limits, entirely unburdened by software abstractions, tracing the pathways pre-sculpted during Trainer Mode.

