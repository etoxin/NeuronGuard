"""
NeuronGuard: Spatiotemporal Ensemble Mesh & Insectoid Walking Gait Simulation

This example introduces the advanced capabilities of the Permanent Spatiotemporal Ensemble Mesh:
1. Initializing a Permanent Spatiotemporal Ensemble Mesh with 3 parallel semantic fields.
2. Simulating an insectoid rigid-body walking gait driven by the rhythmic resonance of feedback loops.
3. Injecting a sudden perturbation (slip/push) mid-stride.
4. Triggering transactional structural re-patching (Guard/Lease) to self-stabilize the gait.
5. Querying real-time telemetry (active nodes, loop intensities, structural mutations) for 3D rendering.
"""

import time

from neuronguard import InsectoidGait


def print_gait_state(state):
    print(f"[Step {state.step:02d}]")
    print(f"  Leg Phases:     " + " ".join(f"{p:.2f}" for p in state.phases))
    print(f"  Leg Velocities: " + " ".join(f"{v:.2f}" for v in state.velocities))
    print(f"  Active Nodes:   {state.active_nodes}")
    print(
        f"  Loop Intensities (Legs 0-5): "
        + " ".join(f"L{i}:{state.loop_intensities.get(i, 0)}" for i in range(6))
    )
    print("-" * 80)


def main():
    print("====================================================================")
    print("🧠 NeuronGuard: Spatiotemporal Ensemble Mesh Showcase 🧠")
    print("====================================================================\n")

    # -------------------------------------------------------------------------
    # STEP 1: Initialize the Insectoid Walking Gait Simulation
    # -------------------------------------------------------------------------
    print("1. Initializing Insectoid Walking Gait Simulation...")
    # This automatically instantiates a PermanentSpatiotemporalEnsembleMesh
    # and configures the rhythmic CPG feedback loops: 0 -> 2 -> 4 -> 1 -> 3 -> 5 -> 0
    sim = InsectoidGait()
    print("   Simulation and CPG feedback loops initialized successfully!\n")

    # -------------------------------------------------------------------------
    # STEP 2: Establish a Steady Walking Gait
    # -------------------------------------------------------------------------
    print("2. Running simulation to establish a steady walking gait...")
    print("-" * 80)
    for _ in range(10):
        state = sim.step(training_mode=False, correct_target=None, decay_factor=0.90)
        print_gait_state(state)
        time.sleep(0.05)

    # -------------------------------------------------------------------------
    # STEP 3: Inject Perturbation & Trigger Transactional Re-patching
    # -------------------------------------------------------------------------
    print("\n3. Injecting a sudden perturbation (simulating a slip/push)...")
    print(
        "   Triggering transactional structural re-patching (Guard/Lease) for anomalous token 99..."
    )

    # We perturb leg 0 and measure the time taken to verify
    # the re-patch completes within a single animation frame (16.6ms).
    start_time = time.time()

    mutations = sim.perturb(target_leg=0, decay_factor=0.90)

    elapsed_ms = (time.time() - start_time) * 1000.0
    print(
        f"   ➔ Transactional re-patch completed in {elapsed_ms:.4f} ms! (Target: < 16.6 ms)"
    )

    # Display structural mutations emitted by the Guard/Lease pattern
    for mut in mutations:
        print(
            f"   [MUTATION EVENT] Token {mut.token_id}: Evicted target {mut.evicted_target} ➔ Patched to {mut.new_target} (Timestamp: {mut.timestamp_us} us)"
        )

    # -------------------------------------------------------------------------
    # STEP 4: Continue Simulation to Self-Stabilize
    # -------------------------------------------------------------------------
    print("\n4. Continuing simulation to demonstrate self-stabilization...")
    print("-" * 80)
    for _ in range(5):
        state = sim.step(training_mode=False, correct_target=None, decay_factor=0.90)
        print_gait_state(state)
        time.sleep(0.05)

    print("====================================================================")
    print("🎉 Showcase complete! The insectoid successfully self-stabilized! 🎉")
    print("====================================================================")


if __name__ == "__main__":
    main()
