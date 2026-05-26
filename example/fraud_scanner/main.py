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

import neuronguard as ng


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
    # STEP 2: Compute Feature Boundaries (Min/Max)
    # -------------------------------------------------------------------------
    print("2. Computing feature boundaries from training set...")
    # Selected features: V10 (0), V12 (1), V14 (2), V17 (3), Amount (4)
    features_min = [float("inf")] * 5
    features_max = [float("-inf")] * 5

    for r in train_records:
        for i in range(5):
            val = r[i]
            if val < features_min[i]:
                features_min[i] = val
            if val > features_max[i]:
                features_max[i] = val

    # -------------------------------------------------------------------------
    # STEP 3: Initialize and Configure the Cortex
    # -------------------------------------------------------------------------
    # We have 5 features, each bucketed into 10 buckets.
    # Total sensory neurons = 5 * 10 = 50.
    # Motor neurons = 2 (0 = Legitimate, 1 = Fraudulent).
    num_sensory = 50
    num_motor = 2

    print("3. Initializing NeuronGuard cortex...")
    field = ng.NeuronGuardField(sensory_count=num_sensory, motor_count=num_motor)

    def get_tokens(record):
        tokens = []
        for i in range(5):
            val = record[i]
            min_val = features_min[i]
            max_val = features_max[i]
            # Map to bucket 0..9
            bucket = 0
            if max_val > min_val:
                if val <= min_val:
                    bucket = 0
                elif val >= max_val:
                    bucket = 9
                else:
                    bucket = int((val - min_val) / (max_val - min_val) * 10)
            tokens.append(i * 10 + bucket)
        return tokens

    # Configure initial baseline: All features start as legitimate (expert 0)
    for i in range(num_sensory):
        field.train_stream([i], correct_motor_id=0, amplify_delta=10, suppress_delta=0)

    # -------------------------------------------------------------------------
    # STEP 4: Train on Real-World Data (Trainer Mode)
    # -------------------------------------------------------------------------
    print("4. Training the cortex on real-world transactions...")
    start_time = time.time()

    for r in train_records:
        tokens = get_tokens(r)
        label = r[5]
        # Train the active tokens to target the correct label (0 or 1)
        # If fraudulent, we amplify with a high delta to override the legitimate baseline.
        # We oversample fraud cases to handle the extreme class imbalance (99.9% legitimate).
        if label == 1:
            for _ in range(100):
                field.train_stream(
                    tokens, correct_motor_id=1, amplify_delta=30, suppress_delta=2
                )
        else:
            field.train_stream(
                tokens, correct_motor_id=0, amplify_delta=2, suppress_delta=10
            )

    duration = time.time() - start_time
    print(f"   Training completed in {duration:.4f}s!\n")

    # -------------------------------------------------------------------------
    # STEP 5: Evaluate Accuracy on Test Set (Run Mode)
    # -------------------------------------------------------------------------
    print("5. Evaluating accuracy on test set...")
    correct_predictions = 0
    confusion_matrix = [[0, 0], [0, 0]]  # [Actual][Predicted]

    for r in test_records:
        tokens = get_tokens(r)
        actual_label = r[5]

        # Reset potentials before presenting the transaction
        field.reset_potentials()

        # Process the stream
        field.process_stream(tokens, training_mode=False)

        # Get the winning expert
        potentials = field.get_potentials()
        predicted_label = potentials.index(max(potentials))

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
