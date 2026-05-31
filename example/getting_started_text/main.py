"""
NeuronGuard: Getting Started - Text Classification

This example introduces the TextClassifier high-level API for text classification,
demonstrating training, atomic inference, and zero-copy mmap loading.
"""

import os
import shutil
import time

from neuronguard import TextClassifier

def main():
    print("====================================================================")
    print("  Getting Started 1: Text Classification & Zero-Copy Loading")
    print("====================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "temp_model_dir")
    
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
    
    print("Training TextClassifier on 6 toy records...")
    t0 = time.perf_counter()
    text_clf.fit_records(records, epochs=5)
    print(f"Training completed in {time.perf_counter() - t0:.4f} seconds.\n")

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

    # Cleanup
    shutil.rmtree(model_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
