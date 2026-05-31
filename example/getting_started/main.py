"""
NeuronGuard: Getting Started Guide

This example introduces the core concepts of NeuronGuard through its Python SDK:
1. TextClassifier: High-level API for text classification.
2. TabularClassifier: High-level API for numerical/tabular data classification.
3. InsectoidGait: High-level API for spatiotemporal ensemble meshes.
4. Raw API: Direct access to the low-level Rust bindings.
"""

import os
import shutil
import tempfile
import time

from neuronguard import TextClassifier, TabularClassifier, InsectoidGait
import neuronguard as ng


def main():
    print("====================================================================")
    print("🧠 Welcome to NeuronGuard: Getting Started Guide 🧠")
    print("====================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "temp_model_dir")
    
    # -------------------------------------------------------------------------
    # SECTION 1: TextClassifier
    # -------------------------------------------------------------------------
    print("--- 1. Text Classification ---")
    print("The TextClassifier handles tokenization, discriminative vocabulary building,")
    print("and multi-epoch training automatically.\n")

    records = [
        (0, 'sports football soccer match game player goal'),
        (0, 'sports basketball court score dunk rebound'),
        (1, 'technology computer software hardware silicon'),
        (1, 'technology processor chip memory circuit'),
        (2, 'science physics biology chemistry atom'),
        (2, 'science galaxy universe planet star'),
    ]

    text_clf = TextClassifier(num_classes=3, vocab_size=20, class_names=['Sports', 'Tech', 'Science'])
    text_clf.fit_records(records, epochs=5)

    test_text = "football soccer match"
    print(f"Input: '{test_text}'")
    print(f"Prediction: {text_clf.predict_name(test_text)}")
    print(f"Raw Scores: {text_clf.predict_scores(test_text)}\n")

    print("Saving and loading models is instant and pointerless...")
    text_clf.save(model_dir)
    loaded_clf = TextClassifier.load(model_dir)
    print(f"Loaded model prediction for '{test_text}': {loaded_clf.predict_name(test_text)}\n")

    # -------------------------------------------------------------------------
    # SECTION 2: TabularClassifier
    # -------------------------------------------------------------------------
    print("--- 2. Tabular Classification ---")
    print("The TabularClassifier automatically buckets continuous features and handles")
    print("class imbalances via weighting.\n")

    tab_clf = TabularClassifier(num_classes=2, num_features=2, buckets_per_feature=5)
    train_data = [
        (1.0, 2.0, 0),
        (1.1, 2.1, 0),
        (8.0, 9.0, 1),
        (8.1, 9.1, 1),
    ]
    
    # Fit with features at indices 0,1 and label at index 2
    tab_clf.fit(train_data, feature_indices=[0, 1], label_index=2, epochs=5)

    test_features = [1.0, 2.0]
    pred = tab_clf.predict(test_features)
    print(f"Input features: {test_features}")
    print(f"Prediction: Class {pred}")
    print(f"Raw Scores: {tab_clf.predict_scores(test_features)}\n")

    # -------------------------------------------------------------------------
    # SECTION 3: InsectoidGait (Spatiotemporal Mesh)
    # -------------------------------------------------------------------------
    print("--- 3. Insectoid Gait (Spatiotemporal Mesh) ---")
    print("A high-level wrapper for exploring CPG feedback loops and topological plasticity.\n")

    sim = InsectoidGait()
    
    print("Running 5 steps to establish a gait...")
    for _ in range(5):
        state = sim.step()
        print(f"Step {state.step:02d} | Phases: {['{:.2f}'.format(p) for p in state.phases]}")
        time.sleep(0.05)

    print("\nInjecting perturbation (slip/push) on leg 0...")
    mutations = sim.perturb(target_leg=0)
    for m in mutations:
        print(f"Mutation: Token {m.token_id} re-patched {m.evicted_target} -> {m.new_target} (at {m.timestamp_us}us)")
    print()

    # -------------------------------------------------------------------------
    # SECTION 4: Raw API (Rust Bindings)
    # -------------------------------------------------------------------------
    print("--- 4. Raw API (Rust Bindings) ---")
    print("The high-level SDK is built on top of the raw Rust bindings, which remain")
    print("available for advanced use cases.\n")

    field = ng.NeuronGuardField(sensory_count=10, motor_count=3)
    field.train_stream([0], 0, 10, 0)
    field.train_stream([1], 1, 15, 0)
    
    field.reset_potentials()
    winner = field.process_stream([0, 1], training_mode=False)
    
    print(f"Raw NeuronGuardField winner for stimuli [0, 1]: {winner} (Expected: 1)")

    # Cleanup
    shutil.rmtree(model_dir, ignore_errors=True)

    print("\n====================================================================")
    print("🎉 Congratulations! You have completed the Getting Started Guide! 🎉")
    print("====================================================================")


if __name__ == "__main__":
    main()
