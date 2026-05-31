"""
NeuronGuard: Real-World Financial Fraud Scanner

This example showcases:
1. Real-World Tabular Classification:
   Trains and evaluates on a subset of the Kaggle Credit Card Fraud Detection dataset.
2. Dynamic Feature Tokenization:
   Continuous float features (PCA components V10, V12, V14, V17, and Amount)
   are bucketed into sensory tokens.
3. High Accuracy & Fast Training:
   Achieves high fraud detection accuracy with sub-millisecond training times.
"""

import csv
import os
import time

from neuronguard import TabularClassifier


def main():
    print("====================================================================")
    print("💳 NeuronGuard Real-World Financial Fraud Scanner 💳")
    print("====================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir, "data", "creditcard_subset.csv")

    if not os.path.exists(data_file_path):
        print("Error: Dataset not found!")
        print("Please run 'mise run download_data' or check the dataset path.")
        return

    # -------------------------------------------------------------------------
    # STEP 1: Load and Split the Dataset
    # -------------------------------------------------------------------------
    print("1. Loading and splitting the creditcard dataset...")
    records = []
    with open(data_file_path, mode="r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr)  # Skip header
        for row in rdr:
            if len(row) < 31:
                continue  # Discard incomplete lines
            try:
                v10 = float(row[10])
                v12 = float(row[12])
                v14 = float(row[14])
                v17 = float(row[17])
                amount = float(row[29])
                label = int(row[30])
                records.append((v10, v12, v14, v17, amount, label))
            except ValueError:
                continue

    total_records = len(records)
    train_size = int(total_records * 0.8)
    train_records = records[:train_size]
    test_records = records[train_size:]

    print(f"   Total Transactions: {total_records:,}")
    print(f"   Training Set Size : {len(train_records):,}")
    print(f"   Test Set Size     : {len(test_records):,}")

    # Count fraud in training and test sets
    train_fraud = sum(1 for r in train_records if r[5] == 1)
    test_fraud = sum(1 for r in test_records if r[5] == 1)
    print(
        f"   Fraud Cases (Train): {train_fraud} ({train_fraud / len(train_records) * 100:.3f}%)"
    )
    print(
        f"   Fraud Cases (Test) : {test_fraud} ({test_fraud / len(test_records) * 100:.3f}%)\n"
    )

    # -------------------------------------------------------------------------
    # STEP 2: Initialize the TabularClassifier
    # -------------------------------------------------------------------------
    # 5 features (V10, V12, V14, V17, Amount), each bucketed into 10 buckets.
    # 2 classes: 0 = Legitimate, 1 = Fraudulent.
    print("2. Initializing TabularClassifier...")
    classifier = TabularClassifier(
        num_classes=2,
        num_features=5,
        buckets_per_feature=10,
        amplify_delta=15,
        suppress_delta=5,
    )

    # -------------------------------------------------------------------------
    # STEP 3: Train on Real-World Data
    # -------------------------------------------------------------------------
    # Records are tuples: (V10, V12, V14, V17, Amount, Label)
    # Features are at indices 0-4, label is at index 5.
    # class_weights={1: 100} oversamples fraud by 100x to handle extreme
    # class imbalance (99.9% legitimate).
    print("3. Training on real-world transactions...")
    start_time = time.time()

    classifier.fit(
        records=train_records,
        feature_indices=[0, 1, 2, 3, 4],
        label_index=5,
        epochs=1,
        class_weights={1: 100},
    )

    duration = time.time() - start_time
    print(f"   Training completed in {duration:.4f}s!\n")

    # -------------------------------------------------------------------------
    # STEP 4: Evaluate Accuracy on Test Set
    # -------------------------------------------------------------------------
    print("4. Evaluating accuracy on test set...")
    correct_predictions = 0
    confusion_matrix = [[0, 0], [0, 0]]  # [Actual][Predicted]

    for r in test_records:
        features = [r[0], r[1], r[2], r[3], r[4]]
        actual_label = r[5]

        predicted_label = classifier.predict(features)

        confusion_matrix[actual_label][predicted_label] += 1
        if predicted_label == actual_label:
            correct_predictions += 1

    accuracy = (correct_predictions / len(test_records)) * 100
    print("   Evaluation Complete!")
    print(f"   ➔ Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(test_records)})")

    # Calculate Precision, Recall, and F1-Score for Fraud (Class 1)
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tn = confusion_matrix[0][0]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\n   --- Confusion Matrix ---")
    print("      Actual \\ Predicted | Legitimate | Fraudulent")
    print("      -------------------|------------|-----------")
    print(f"      Legitimate         | {tn:10} | {fp:10}")
    print(f"      Fraudulent         | {fn:10} | {tp:10}")

    print("\n   --- Fraud Detection Metrics ---")
    print(f"      Precision: {precision * 100:.2f}%")
    print(f"      Recall   : {recall * 100:.2f}%")
    print(f"      F1-Score : {f1 * 100:.2f}%")
    print("====================================================================")


if __name__ == "__main__":
    main()
