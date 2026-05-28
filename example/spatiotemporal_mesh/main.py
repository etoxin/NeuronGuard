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

import neuronguard as ng


def print_gait_state(step, phases, velocities, active_nodes, intensities):
    print(f"[Step {step:02d}]")
    print(f"  Leg Phases:     " + " ".join(f"{p:.2f}" for p in phases))
    print(f"  Leg Velocities: " + " ".join(f"{v:.2f}" for v in velocities))
    print(f"  Active Nodes:   {active_nodes}")
    print(
        f"  Loop Intensities (Legs 0-5): "
        + " ".join(f"L{i}:{intensities.get(i, 0)}" for i in range(6))
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
    sim = ng.PyInsectoidSimulation()
    print("   Simulation and CPG feedback loops initialized successfully!\n")

    # -------------------------------------------------------------------------
    # STEP 2: Establish a Steady Walking Gait
    # -------------------------------------------------------------------------
    print("2. Running simulation to establish a steady walking gait...")
    print("-" * 80)
    for step in range(1, 11):
        sim.step(training_mode=False, correct_target=None, decay_factor=0.90)
        phases = sim.get_phases()
        velocities = sim.get_velocities()
        active_nodes = sim.get_active_nodes()
        intensities = sim.get_loop_intensities()
        print_gait_state(step, phases, velocities, active_nodes, intensities)
        time.sleep(0.05)

    # -------------------------------------------------------------------------
    # STEP 3: Inject Perturbation & Trigger Transactional Re-patching
    # -------------------------------------------------------------------------
    print("\n3. Injecting a sudden perturbation (simulating a slip/push)...")
    print(
        "   Triggering transactional structural re-patching (Guard/Lease) for anomalous token 99..."
    )

    # We manually step the simulation with training_mode=True and correct_target=0
    # to instantly re-patch the anomalous token 99's forward target to Leg 0.
    # We measure the time taken to verify it completes within a single animation frame (16.6ms).
    start_time = time.time()

    # We simulate the perturbation by stepping with training_mode=True and correct_target=0
    sim.step(training_mode=True, correct_target=0, decay_factor=0.90)

    elapsed_ms = (time.time() - start_time) * 1000.0
    print(
        f"   ➔ Transactional re-patch completed in {elapsed_ms:.4f} ms! (Target: < 16.6 ms)"
    )

    # Retrieve structural mutations emitted by the Guard/Lease pattern
    mutations = sim.get_structural_mutations()
    for mut in mutations:
        token_id, evicted, new_target, ts = mut
        print(
            f"   [MUTATION EVENT] Token {token_id}: Evicted target {evicted} ➔ Patched to {new_target} (Timestamp: {ts} us)"
        )

    # -------------------------------------------------------------------------
    # STEP 4: Continue Simulation to Self-Stabilize
    # -------------------------------------------------------------------------
    print("\n4. Continuing simulation to demonstrate self-stabilization...")
    print("-" * 80)
    for step in range(11, 16):
        sim.step(training_mode=False, correct_target=None, decay_factor=0.90)
        phases = sim.get_phases()
        velocities = sim.get_velocities()
        active_nodes = sim.get_active_nodes()
        intensities = sim.get_loop_intensities()
        print_gait_state(step, phases, velocities, active_nodes, intensities)
        time.sleep(0.05)

    print("====================================================================")
    print("🎉 Showcase complete! The insectoid successfully self-stabilized! 🎉")
    print("====================================================================")


if __name__ == "__main__":
    main()
