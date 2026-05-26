"""
NeuronGuard: Getting Started Guide

This example introduces the core concepts of NeuronGuard:
1. Initializing a NeuronGuardField (a neuromorphic cortex).
2. Configuring sensory-to-motor connections.
3. Running real-time inference (Run Mode) by processing sensory stimuli streams.
4. Performing supervised learning (Trainer Mode) using the Guard/Lease pattern.
5. Saving and loading model weights instantly.
"""

import os

import neuronguard as ng


def main():
    print("====================================================================")
    print("🧠 Welcome to NeuronGuard: Getting Started Guide 🧠")
    print("====================================================================\n")

    # -------------------------------------------------------------------------
    # STEP 1: Initialize the Neuromorphic Cortex
    # -------------------------------------------------------------------------
    # We define a small cortex with 10 sensory neurons and 3 motor neurons.
    # Sensory neurons represent incoming stimuli (e.g., words, pixels, sensor inputs).
    # Motor neurons represent output categories or actions (e.g., classifications).
    num_sensory = 10
    num_motor = 3

    print(
        f"1. Initializing cortex with {num_sensory} sensory and {num_motor} motor neurons..."
    )
    cortex = ng.NeuronGuardField(sensory_count=num_sensory, motor_count=num_motor)
    print("   Cortex allocated successfully in CPU L1/L2 cache!\n")

    # -------------------------------------------------------------------------
    # STEP 2: Configure Initial Connections (Run Mode)
    # -------------------------------------------------------------------------
    # Let's manually connect some sensory neurons to motor neurons.
    # Sensory Neuron 0 targets Motor Neuron 0 with a weight of 10.
    # Sensory Neuron 1 targets Motor Neuron 1 with a weight of 15.
    print("2. Configuring initial sensory-to-motor connections...")
    cortex.train_stream(
        sensory_tokens=[0], correct_motor_id=0, amplify_delta=10, suppress_delta=0
    )
    cortex.train_stream(
        sensory_tokens=[1], correct_motor_id=1, amplify_delta=15, suppress_delta=0
    )
    print("   Initial connections configured.\n")

    # -------------------------------------------------------------------------
    # STEP 3: Run Real-Time Inference (Run Mode)
    # -------------------------------------------------------------------------
    # We present a stream of active sensory stimuli to the cortex.
    # The cortex drops the Python GIL and processes the stream in parallel
    # across background worker threads, accumulating potentials on the motor neurons.
    print("3. Running real-time inference...")
    active_stimuli = [0, 1]  # Stimuli 0 and 1 are active simultaneously

    # Reset potentials before presenting the new stream
    cortex.reset_potentials()

    # Process the stream (training_mode=False)
    # This returns the index of the motor neuron with the highest accumulated potential.
    winning_motor_id = cortex.process_stream(active_stimuli, training_mode=False)
    potentials = cortex.get_potentials()

    print(f"   Active Stimuli: {active_stimuli}")
    print(f"   Motor Potentials: {potentials}")
    print(
        f"   ➔ Winner: Motor Neuron {winning_motor_id} (Expected: 1, because weight 15 > 10)\n"
    )

    # -------------------------------------------------------------------------
    # STEP 4: Supervised Learning (Trainer Mode)
    # -------------------------------------------------------------------------
    # Let's train the cortex to associate Sensory Neuron 2 with Motor Neuron 2.
    # We use `train_stream` which implements the Guard/Lease pattern:
    # - Amplifies the correct pathway.
    # - Suppresses incorrect/competing pathways.
    print("4. Training the cortex (Trainer Mode)...")
    print("   Associating Sensory Neuron 2 with Motor Neuron 2...")

    # Train Sensory Neuron 2 to target Motor Neuron 2
    cortex.train_stream(
        sensory_tokens=[2], correct_motor_id=2, amplify_delta=20, suppress_delta=5
    )

    # Let's verify the training worked!
    cortex.reset_potentials()
    winning_motor_id = cortex.process_stream([2], training_mode=False)
    potentials = cortex.get_potentials()
    print("   Active Stimuli: [2]")
    print(f"   Motor Potentials: {potentials}")
    print(f"   ➔ Winner: Motor Neuron {winning_motor_id} (Expected: 2)\n")

    # -------------------------------------------------------------------------
    # STEP 5: Save and Load Model Weights
    # -------------------------------------------------------------------------
    # NeuronGuard supports instant, pointerless serialization of model weights.
    print("5. Saving and loading model weights...")
    weights_path = "getting_started_weights.bin"

    # Save weights to disk
    cortex.save_weights(weights_path)
    print(f"   Weights saved successfully to '{weights_path}'")

    # Create a brand new, empty cortex
    new_cortex = ng.NeuronGuardField(sensory_count=num_sensory, motor_count=num_motor)

    # Load the saved weights into the new cortex
    new_cortex.load_weights(weights_path)
    print("   Weights loaded successfully into a new cortex in < 1ms!")

    # Verify the new cortex has the same behavior
    new_cortex.reset_potentials()
    winning_motor_id = new_cortex.process_stream([2], training_mode=False)
    print(f"   ➔ Winner in New Cortex: Motor Neuron {winning_motor_id} (Expected: 2)\n")

    # Clean up the weights file
    if os.path.exists(weights_path):
        os.remove(weights_path)

    print("====================================================================")
    print("🎉 Congratulations! You have completed the Getting Started Guide! 🎉")
    print("====================================================================")


if __name__ == "__main__":
    main()
