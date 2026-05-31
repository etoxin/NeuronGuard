"""
NeuronGuard: Getting Started - Parallel Batch Processing

This example dives into the raw Rust bindings to demonstrate true GIL-free
multi-core parallel processing using Rayon.
"""

import time
import neuronguard as ng

def main():
    print("====================================================================")
    print("  Getting Started 3: Parallel Batch Processing API")
    print("====================================================================\n")

    print("The high-level SDK is built on top of the raw Rust bindings, which provide")
    print("true GIL-free multi-core parallel processing using Rayon.\n")

    field = ng.NeuronGuardField(sensory_count=10, motor_count=3)
    
    print("Training 100,000 samples across all CPU cores...")
    # Generate 100k dummy training tasks: (tokens, label)
    batch_train_tasks = [([0, 1, 2], 0)] * 50_000 + [([3, 4, 5], 1)] * 50_000
    
    t0 = time.perf_counter()
    # Batch train with amplify=10, suppress=0
    field.train_batch(batch_train_tasks, 10, 0)
    print(f"Batch training completed in {time.perf_counter() - t0:.4f} seconds.\n")

    print("Predicting 100,000 samples across all CPU cores...")
    # Predict 100k times
    batch_predict_tasks = [[0, 1, 2]] * 50_000 + [[3, 4, 5]] * 50_000
    
    t0 = time.perf_counter()
    results = field.predict_batch(batch_predict_tasks)
    print(f"Batch inference completed in {time.perf_counter() - t0:.4f} seconds.")
    print(f"First result: Class {results[0]}, 50,001st result: Class {results[50000]}")

if __name__ == "__main__":
    main()
