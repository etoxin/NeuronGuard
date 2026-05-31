import os
import time
from neuronguard import TabularClassifier

def main():
    print("====================================================================")
    print("📦 Higgs Boson (11 Million Samples) Tabular PoC (Python) 📦")
    print("====================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(script_dir, "data", "train.csv")
    test_csv = os.path.join(script_dir, "data", "test.csv")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("Dataset not found! Run `mise run examples:higgs_boson:download_data` first.")
        return

    # 2 Classes (Signal=1, Background=0), 28 features. 
    # We use 100 buckets per feature to handle continuous physics data cleanly.
    # Feature interactions are CRITICAL for physics data to learn 2D non-linear patterns.
    classifier = TabularClassifier(
        num_classes=2,
        num_features=28,
        buckets_per_feature=100,
        amplify_delta=15,
        suppress_delta=5,
        use_feature_interactions=True,
        interaction_vocab_size=5000000
    )

    feature_indices = list(range(1, 29))
    label_index = 0

    print("--- Streaming Training on 10,000,000 Samples ---")
    print("  (Using O(1) Memory via streaming generator)")
    start_time = time.time()
    
    classifier.fit_from_csv(
        file_path=train_csv,
        feature_indices=feature_indices,
        label_index=label_index,
        epochs=1
    )
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s!\n")

    print("--- Streaming Evaluation on 1,000,000 Test Samples ---")
    start_time = time.time()
    
    accuracy, report = classifier.evaluate_from_csv(
        file_path=test_csv,
        feature_indices=feature_indices,
        label_index=label_index,
    )
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f}s!\n")

    print(report)
    print("====================================================================")

if __name__ == "__main__":
    main()
