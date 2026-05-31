"""
NeuronGuard: Amazon Reviews Polarity (3.6 Million Samples)

This example proves NeuronGuard's extreme scalability and speed.
It trains on a massive 3.6 million row dataset (which would take hours
in traditional Python ML frameworks without a GPU) entirely on the CPU
in a matter of minutes, fitting completely in CPU L1/L2 cache.
"""

import os
import time
from neuronguard import TextClassifier

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "data", "amazon_review_polarity_csv", "train.csv")
    test_path = os.path.join(script_dir, "data", "amazon_review_polarity_csv", "test.csv")

    if not os.path.exists(train_path):
        print("Error: Dataset not found!")
        print("Please run `mise run examples:amazon_reviews:download_data` first.")
        return

    print("====================================================================")
    print("📦 Amazon Reviews Polarity (3,600,000 Samples) PoC (Python) 📦")
    print("====================================================================\n")

    # The dataset has 2 classes: 1 (Negative) and 2 (Positive). 
    # label_offset=-1 will map them to 0 and 1.
    classifier = TextClassifier(
        num_classes=2,
        vocab_size=15000,
        class_names=["Negative", "Positive"],
    )

    # --- Training --------------------------------------------------------
    print("--- Training on 3,600,000 Samples (1 epoch) ---")
    print("Parsing 3.6M CSV rows and compiling vocabulary. This may take a minute...")

    start_time = time.time()
    # We use 1 epoch to demonstrate raw ingestion speed.
    classifier.fit(train_path, text_col=[1, 2], label_col=0, epochs=1)
    duration = time.time() - start_time

    print(f"Training completed in {duration:.2f}s!\n")

    # --- Evaluation ------------------------------------------------------
    print("--- Evaluating on 400,000 Test Samples ---")
    
    start_time = time.time()
    accuracy, report = classifier.evaluate(test_path, text_col=[1, 2], label_col=0)
    eval_duration = time.time() - start_time
    
    print(f"Evaluation completed in {eval_duration:.2f}s!\n")
    print(report)
    print("====================================================================")

if __name__ == "__main__":
    main()
