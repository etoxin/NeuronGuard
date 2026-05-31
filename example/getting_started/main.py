"""
NeuronGuard: Getting Started Guide

This example introduces the core concepts of NeuronGuard through its Python SDK:
1. TextClassifier: High-level API for text classification.
2. TabularClassifier: High-level API for numerical/tabular data classification.
3. Batch Processing: True multi-core parallel batch processing.
"""

import os
import shutil
import time

from neuronguard import TextClassifier, TabularClassifier
import neuronguard as ng

def main():
    print("====================================================================")
    print("  Welcome to NeuronGuard: Getting Started Guide")
    print("====================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "temp_model_dir")
    
    # -------------------------------------------------------------------------
    # SECTION 1: TextClassifier & Zero-Copy Mmap
    # -------------------------------------------------------------------------
    print("--- 1. Text Classification & Zero-Copy Loading ---")
    print("The TextClassifier handles tokenization (powered by native Rust),")
    print("discriminative vocabulary building, and multi-epoch training.\n")

    records = [
        (0, 'sports football soccer match game player goal'),
        (0, 'sports basketball court score dunk rebound'),
        (1, 'technology computer software hardware silicon'),
        (1, 'technology processor chip memory circuit'),
        (2, 'science physics biology chemistry atom'),
        (2, 'science galaxy universe planet star'),
    ]

    text_clf = TextClassifier(num_classes=3, vocab_size=20, class_names=['Sports', 'Tech', 'Science'])
    t0 = time.perf_counter()
    text_clf.fit_records(records, epochs=5)
    print(f"Training completed in {time.perf_counter() - t0:.4f} seconds.")

    test_text = "football soccer match"
    print(f"Input: '{test_text}'")
    print(f"Prediction: {text_clf.predict_name(test_text)}")
    print(f"Raw Scores: {text_clf.predict_scores(test_text)}\n")

    print("Saving model to disk...")
    text_clf.save(model_dir)
    
    print("Loading model via Zero-Copy Memory Map (mmap)...")
    t0 = time.perf_counter()
    loaded_clf = TextClassifier.load(model_dir)
    print(f"Model loaded in {time.perf_counter() - t0:.6f} seconds!")
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
    # SECTION 3: Parallel Batch Processing API (Rust Core)
    # -------------------------------------------------------------------------
    print("--- 3. Parallel Batch Processing API ---")
    print("The high-level SDK is built on top of the raw Rust bindings, which provide")
    print("true GIL-free multi-core parallel processing using Rayon.\n")

    field = ng.NeuronGuardField(sensory_count=10, motor_count=3)
    
    # Train heavily using the batch API
    print("Training 100,000 samples across all CPU cores...")
    # Generate 100k dummy training tasks: (tokens, label)
    batch_train_tasks = [([0, 1, 2], 0)] * 50_000 + [([3, 4, 5], 1)] * 50_000
    
    t0 = time.perf_counter()
    field.train_batch(batch_train_tasks, 10, 0)
    print(f"Batch training completed in {time.perf_counter() - t0:.4f} seconds.\n")

    print("Predicting 100,000 samples across all CPU cores...")
    # Predict 100k times
    batch_predict_tasks = [[0, 1, 2]] * 50_000 + [[3, 4, 5]] * 50_000
    
    t0 = time.perf_counter()
    results = field.predict_batch(batch_predict_tasks)
    print(f"Batch inference completed in {time.perf_counter() - t0:.4f} seconds.")
    print(f"First result: Class {results[0]}, 50,001st result: Class {results[50000]}")

    # Cleanup
    shutil.rmtree(model_dir, ignore_errors=True)

    print("\n====================================================================")
    print("  Congratulations! You have completed the Getting Started Guide!  ")
    print("====================================================================")


if __name__ == "__main__":
    main()
