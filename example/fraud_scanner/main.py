"""
NeuronGuard: Real-World Financial Fraud Scanner

This example showcases:
1. Real-World Tabular Classification:
   Trains and evaluates on the complete Kaggle Credit Card Fraud Detection dataset.
2. Dynamic Feature Tokenization:
   All 28 PCA components plus Amount are bucketed into sensory tokens.
3. Imbalance-Aware Evaluation:
   Uses stratified cross-validation and PR-AUC alongside threshold metrics.
"""

import argparse
import csv
import os
import random
import time

from neuronguard import TabularClassifier


def make_stratified_folds(records, n_splits=5, seed=42):
    """Return deterministic folds with each class distributed across every fold."""
    records_by_class = {}
    for record in records:
        records_by_class.setdefault(record[-1], []).append(record)

    rng = random.Random(seed)
    folds = [[] for _ in range(n_splits)]
    for class_records in records_by_class.values():
        rng.shuffle(class_records)
        for index, record in enumerate(class_records):
            folds[index % n_splits].append(record)

    for fold in folds:
        rng.shuffle(fold)
    return folds


def precision_recall_auc(labels, scores):
    """Calculate step-wise PR-AUC (average precision), handling tied scores together."""
    positive_count = sum(labels)
    if positive_count == 0:
        return 0.0

    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    index = 0

    while index < len(ranked):
        score = ranked[index][0]
        group_true_positives = 0
        group_false_positives = 0
        while index < len(ranked) and ranked[index][0] == score:
            if ranked[index][1] == 1:
                group_true_positives += 1
            else:
                group_false_positives += 1
            index += 1

        true_positives += group_true_positives
        false_positives += group_false_positives
        recall = true_positives / positive_count
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall

    return area


def predict_with_jitter(
    classifier,
    features,
    baseline_scores,
    feature_mins,
    feature_maxs,
    bucket_widths,
    jitter_samples,
    jitter_fraction,
    rng,
):
    """Average scores from the original input and bucket-relative perturbations."""
    score_totals = [float(score) for score in baseline_scores]
    for _ in range(jitter_samples):
        perturbed = []
        for value, minimum, maximum, bucket_width in zip(
            features, feature_mins, feature_maxs, bucket_widths
        ):
            offset = rng.uniform(-jitter_fraction, jitter_fraction) * bucket_width
            perturbed.append(min(max(value + offset, minimum), maximum))

        scores = classifier.predict_scores(perturbed)
        for class_index, score in enumerate(scores):
            score_totals[class_index] += score

    prediction_count = jitter_samples + 1
    return [score / prediction_count for score in score_totals]


def report_metrics(title, confusion_matrix, total_records, mean_pr_auc):
    """Print aggregate binary-classification metrics and return them as a dict."""
    true_positives = confusion_matrix[1][1]
    false_positives = confusion_matrix[0][1]
    false_negatives = confusion_matrix[1][0]
    true_negatives = confusion_matrix[0][0]
    correct_predictions = true_negatives + true_positives

    accuracy = correct_predictions / total_records
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print(f"\n   --- {title} ---")
    print(f"   ➔ Accuracy: {accuracy * 100:.2f}% ({correct_predictions}/{total_records})")
    print("      Actual \\ Predicted | Legitimate | Fraudulent")
    print("      -------------------|------------|-----------")
    print(f"      Legitimate         | {true_negatives:10} | {false_positives:10}")
    print(f"      Fraudulent         | {false_negatives:10} | {true_positives:10}")
    print(f"      Precision: {precision * 100:.2f}%")
    print(f"      Recall   : {recall * 100:.2f}%")
    print(f"      F1-Score : {f1 * 100:.2f}%")
    print(f"      Mean PR-AUC: {mean_pr_auc * 100:.2f}%")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": mean_pr_auc,
    }


def main(epochs=1, jitter_samples=0, jitter_fraction=0.1):
    print("====================================================================")
    print("💳 NeuronGuard Real-World Financial Fraud Scanner 💳")
    print("====================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir, "data", "creditcard.csv")

    if not os.path.exists(data_file_path):
        print("Error: Dataset not found!")
        print("Please run 'mise run examples:fraud_scanner:download_data' or check the dataset path.")
        return

    # -------------------------------------------------------------------------
    # STEP 1: Load the Dataset
    # -------------------------------------------------------------------------
    print("1. Loading the creditcard dataset...")
    records = []
    with open(data_file_path, mode="r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr)  # Skip header
        for row in rdr:
            if len(row) < 31:
                continue  # Discard incomplete lines
            try:
                features = [float(value) for value in row[1:29]]
                features.append(float(row[29]))  # Amount
                label = int(row[30])
                records.append((*features, label))
            except ValueError:
                continue

    total_records = len(records)
    num_features = 29
    feature_indices = list(range(num_features))
    label_index = num_features
    fraud_records = sum(1 for record in records if record[label_index] == 1)
    print(f"   Total Transactions: {total_records:,}")
    print(
        f"   Fraud Cases       : {fraud_records} "
        f"({fraud_records / total_records * 100:.3f}%)\n"
    )

    # -------------------------------------------------------------------------
    # STEP 2: Build deterministic stratified folds
    # -------------------------------------------------------------------------
    n_splits = 5
    folds = make_stratified_folds(records, n_splits=n_splits, seed=42)
    fold_fraud_counts = [
        sum(1 for record in fold if record[label_index] == 1) for fold in folds
    ]
    print(f"2. Building {n_splits} deterministic stratified folds...")
    print(f"   Fraud cases per fold: {fold_fraud_counts}\n")

    # -------------------------------------------------------------------------
    # STEP 3: Train and evaluate every fold
    # -------------------------------------------------------------------------
    # Records are tuples: (V1, ..., V28, Amount, Label).
    # Features are at indices 0-28, label is at index 29.
    # class_weights={1: 100} oversamples fraud by 100x to handle extreme
    # class imbalance (99.9% legitimate).
    print(f"3. Running stratified cross-validation ({epochs} training epoch{'s' if epochs != 1 else ''})...")
    baseline_confusion = [[0, 0], [0, 0]]  # [Actual][Predicted]
    jitter_confusion = [[0, 0], [0, 0]]
    training_durations = []
    baseline_fold_pr_aucs = []
    jitter_fold_pr_aucs = []

    for fold_index, test_records in enumerate(folds):
        validation_index = (fold_index + 1) % n_splits
        validation_records = folds[validation_index]
        train_records = [
            record
            for index, fold in enumerate(folds)
            if index not in (fold_index, validation_index)
            for record in fold
        ]

        # 29 features (V1-V28 and Amount), each quantile-bucketed into 16 buckets.
        # 2 classes: 0 = Legitimate, 1 = Fraudulent.
        classifier = TabularClassifier(
            num_classes=2,
            num_features=num_features,
            buckets_per_feature=16,
            amplify_delta=15,
            suppress_delta=5,
            baseline_delta=0,
            bucket_strategy="quantile",
        )

        start_time = time.perf_counter()
        classifier.fit(
            records=train_records,
            feature_indices=feature_indices,
            label_index=label_index,
            epochs=epochs,
            shuffle=False,
            class_weights={1: 100},
        )
        threshold = classifier.tune_decision_threshold(
            validation_records,
            feature_indices=feature_indices,
            label_index=label_index,
        )
        training_duration = time.perf_counter() - start_time
        training_durations.append(training_duration)

        feature_mins = classifier._features_min
        feature_maxs = classifier._features_max
        bucket_widths = [
            (maximum - minimum) / classifier.buckets_per_feature
            for minimum, maximum in zip(feature_mins, feature_maxs)
        ]
        jitter_rng = random.Random(10_000 + fold_index)

        baseline_true_positives = 0
        jitter_true_positives = 0
        fold_labels = []
        baseline_fold_scores = []
        jitter_fold_scores = []
        for record in test_records:
            features = list(record[:num_features])
            actual_label = record[label_index]
            baseline_scores = classifier.predict_scores(features)
            baseline_margin = baseline_scores[1] - baseline_scores[0]
            baseline_prediction = (
                1 if baseline_margin >= classifier.decision_threshold else 0
            )

            baseline_confusion[actual_label][baseline_prediction] += 1
            fold_labels.append(actual_label)
            baseline_fold_scores.append(baseline_margin)
            if actual_label == 1 and baseline_prediction == 1:
                baseline_true_positives += 1

            if jitter_samples:
                jitter_scores = predict_with_jitter(
                    classifier,
                    features,
                    baseline_scores,
                    feature_mins,
                    feature_maxs,
                    bucket_widths,
                    jitter_samples,
                    jitter_fraction,
                    jitter_rng,
                )
                jitter_margin = jitter_scores[1] - jitter_scores[0]
                jitter_prediction = (
                    1 if jitter_margin >= classifier.decision_threshold else 0
                )
                jitter_confusion[actual_label][jitter_prediction] += 1
                jitter_fold_scores.append(jitter_margin)
                if actual_label == 1 and jitter_prediction == 1:
                    jitter_true_positives += 1

        fold_fraud = fold_fraud_counts[fold_index]
        baseline_pr_auc = precision_recall_auc(fold_labels, baseline_fold_scores)
        baseline_fold_pr_aucs.append(baseline_pr_auc)
        fold_summary = (
            f"   Fold {fold_index + 1}: trained on {len(train_records):,}, "
            f"tuned on {len(validation_records):,}, tested on {len(test_records):,}, "
            f"threshold {threshold:.1f}, detected "
            f"{baseline_true_positives}/{fold_fraud} frauds, "
            f"PR-AUC {baseline_pr_auc * 100:.2f}%"
        )
        if jitter_samples:
            jitter_pr_auc = precision_recall_auc(fold_labels, jitter_fold_scores)
            jitter_fold_pr_aucs.append(jitter_pr_auc)
            fold_summary += (
                f"; jitter detected {jitter_true_positives}/{fold_fraud}, "
                f"PR-AUC {jitter_pr_auc * 100:.2f}%"
            )
        print(f"{fold_summary}, trained in {training_duration:.4f}s")

    print(
        f"   Mean training time: {sum(training_durations) / n_splits:.4f}s "
        f"({sum(training_durations):.4f}s total)\n"
    )

    # -------------------------------------------------------------------------
    # STEP 4: Report aggregate out-of-fold metrics
    # -------------------------------------------------------------------------
    print("4. Aggregate out-of-fold evaluation...")
    baseline_mean_pr_auc = sum(baseline_fold_pr_aucs) / len(baseline_fold_pr_aucs)
    report_metrics(
        "Normal prediction",
        baseline_confusion,
        total_records,
        baseline_mean_pr_auc,
    )
    if jitter_samples:
        jitter_mean_pr_auc = sum(jitter_fold_pr_aucs) / len(jitter_fold_pr_aucs)
        report_metrics(
            f"Jitter ensemble ({jitter_samples} altered + original)",
            jitter_confusion,
            total_records,
            jitter_mean_pr_auc,
        )
    print("====================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NeuronGuard fraud example.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes over each training fold (default: 1).",
    )
    parser.add_argument(
        "--jitter-samples",
        type=int,
        default=0,
        help="Number of altered predictions to average with the original (default: 0).",
    )
    parser.add_argument(
        "--jitter-fraction",
        type=float,
        default=0.1,
        help="Maximum alteration as a fraction of each feature's bucket width (default: 0.1).",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.jitter_samples < 0:
        parser.error("--jitter-samples cannot be negative")
    if not 0.0 <= args.jitter_fraction <= 1.0:
        parser.error("--jitter-fraction must be between 0 and 1")
    main(
        epochs=args.epochs,
        jitter_samples=args.jitter_samples,
        jitter_fraction=args.jitter_fraction,
    )
