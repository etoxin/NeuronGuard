"""Reproducible NeuronGuard comparison against three classical baselines.

Run from the repository root:

    .venv/bin/python benchmarks/compare_models.py

The parent process launches one worker per dataset/model pair so peak RSS is not
contaminated by models benchmarked earlier in the same process.
"""

import argparse
import json
import pickle
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from neuronguard import TabularClassifier


ROOT = Path(__file__).resolve().parents[1]
FRAUD_PATH = ROOT / "example" / "fraud_scanner" / "data" / "creditcard.csv"
DATASETS = ("credit-card-fraud", "breast-cancer", "wine")
MODELS = ("neuronguard", "naive-bayes", "logistic-regression", "boosted-tree")


def load_dataset(name):
    if name == "credit-card-fraud":
        if not FRAUD_PATH.exists():
            raise FileNotFoundError(
                "Fraud data is missing; run the fraud example's download_data task first"
            )
        data = np.loadtxt(
            FRAUD_PATH,
            delimiter=",",
            skiprows=1,
            usecols=range(1, 31),
            dtype=np.float64,
        )
        return data[:, :-1], data[:, -1].astype(np.int64)
    if name == "breast-cancer":
        dataset = load_breast_cancer()
        return dataset.data.astype(np.float64), dataset.target.astype(np.int64)
    if name == "wine":
        dataset = load_wine()
        return dataset.data.astype(np.float64), dataset.target.astype(np.int64)
    raise ValueError(f"unknown dataset: {name}")


def split_dataset(features, labels):
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_x,
        train_y,
        test_size=0.25,
        random_state=42,
        stratify=train_y,
    )
    return train_x, validation_x, test_x, train_y, validation_y, test_y


def peak_rss_during(operation):
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.wait(0.001):
            peak = max(peak, process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        result = operation()
    finally:
        stop.set()
        sampler.join()
        peak = max(peak, process.memory_info().rss)
    return result, max(0, peak - baseline)


def fit_neuronguard(train_x, train_y, validation_x, validation_y):
    class_count = int(np.max(train_y)) + 1
    feature_count = train_x.shape[1]
    counts = np.bincount(train_y, minlength=class_count)
    largest_class = int(np.max(counts))
    class_weights = {
        class_index: min(100, max(1, round(largest_class / count)))
        for class_index, count in enumerate(counts)
        if count
    }
    model = TabularClassifier(
        num_classes=class_count,
        num_features=feature_count,
        # Small-data ablation: eight equal-width bins with light smoothing
        # preserve the associative representation while avoiding the severe
        # over-regularisation of 16 quantile bins + Laplace alpha=1.  This
        # configuration is fixed in the worker. The repository's historical
        # split is a development benchmark; release claims should add nested
        # repeated cross-validation rather than treating this split as unseen.
        buckets_per_feature=8,
        bucket_strategy="uniform",
        baseline_delta=0,
        smoothing=0.01,
    )
    model.fit_xy(
        train_x,
        train_y,
        epochs=1,
        shuffle=False,
        class_weights=class_weights,
    )
    if class_count == 2:
        model.tune_decision_threshold_xy(
            validation_x, validation_y, metric="accuracy"
        )
    return model


def make_baseline(name, class_count):
    if name == "naive-bayes":
        return GaussianNB()
    if name == "logistic-regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                random_state=42,
            ),
        )
    if name == "boosted-tree":
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=100,
            random_state=42,
        )
    raise ValueError(f"unknown baseline: {name}")


def model_size_bytes(model, model_name):
    if model_name != "neuronguard":
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)
        return sum(path.stat().st_size for path in Path(directory).iterdir())


def predict_neuronguard(model, features):
    predictions = []
    scores = []
    for row in features:
        row_values = row.tolist()
        raw_scores = model.predict_scores(row_values)
        if model.num_classes == 2:
            margin = raw_scores[1] - raw_scores[0]
            predictions.append(1 if margin >= model.decision_threshold else 0)
            scores.append(margin)
        else:
            predictions.append(int(np.argmax(raw_scores)))
            scale = model.weight_scale
            shifted = np.asarray(raw_scores, dtype=float) / scale
            shifted -= np.max(shifted)
            probabilities = np.exp(shifted)
            scores.append(probabilities / np.sum(probabilities))
    return np.asarray(predictions), np.asarray(scores)


def predict_baseline(model, features, class_count):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    scores = probabilities[:, 1] if class_count == 2 else probabilities
    return predictions, scores


def pr_auc(labels, scores, class_count):
    if class_count == 2:
        return float(average_precision_score(labels, scores))
    binary_labels = label_binarize(labels, classes=np.arange(class_count))
    return float(average_precision_score(binary_labels, scores, average="macro"))


def median_single_latency_us(model, model_name, features, sample_count=500):
    sample = features[: min(sample_count, len(features))]
    latencies = []
    for row in sample[:10]:
        if model_name == "neuronguard":
            model.predict(row.tolist())
        else:
            model.predict(row.reshape(1, -1))
    for row in sample:
        start = time.perf_counter_ns()
        if model_name == "neuronguard":
            model.predict(row.tolist())
        else:
            model.predict(row.reshape(1, -1))
        latencies.append((time.perf_counter_ns() - start) / 1_000)
    return float(statistics.median(latencies))


def run_worker(dataset_name, model_name):
    features, labels = load_dataset(dataset_name)
    splits = split_dataset(features, labels)
    train_x, validation_x, test_x, train_y, validation_y, test_y = splits
    class_count = int(np.max(labels)) + 1

    def train():
        start = time.perf_counter()
        if model_name == "neuronguard":
            model = fit_neuronguard(train_x, train_y, validation_x, validation_y)
        else:
            model = make_baseline(model_name, class_count)
            model.fit(train_x, train_y)
        return model, time.perf_counter() - start

    (model, training_seconds), peak_rss_bytes = peak_rss_during(train)
    inference_start = time.perf_counter()
    if model_name == "neuronguard":
        predictions, scores = predict_neuronguard(model, test_x)
    else:
        predictions, scores = predict_baseline(model, test_x, class_count)
    batch_inference_seconds = time.perf_counter() - inference_start

    return {
        "dataset": dataset_name,
        "model": model_name,
        "samples": int(len(features)),
        "features": int(features.shape[1]),
        "classes": class_count,
        "accuracy": float(accuracy_score(test_y, predictions)),
        "pr_auc": pr_auc(test_y, scores, class_count),
        "training_seconds": training_seconds,
        "inference_us_per_sample": batch_inference_seconds / len(test_x) * 1_000_000,
        "single_latency_us_p50": median_single_latency_us(
            model, model_name, test_x
        ),
        "peak_training_rss_mb": peak_rss_bytes / (1024 * 1024),
        "model_size_kb": model_size_bytes(model, model_name) / 1024,
    }


def print_markdown(results):
    print(
        "| Dataset | Model | Accuracy | PR-AUC | Train (s) | Batch µs/sample | "
        "Single p50 (µs) | Peak train MB | Model KB |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['dataset']} | {result['model']} | "
            f"{result['accuracy']:.4f} | {result['pr_auc']:.4f} | "
            f"{result['training_seconds']:.4f} | "
            f"{result['inference_us_per_sample']:.2f} | "
            f"{result['single_latency_us_p50']:.2f} | "
            f"{result['peak_training_rss_mb']:.2f} | "
            f"{result['model_size_kb']:.2f} |"
        )


def run_parent():
    results = []
    for dataset_name in DATASETS:
        for model_name in MODELS:
            command = [
                sys.executable,
                __file__,
                "--worker",
                "--dataset",
                dataset_name,
                "--model",
                model_name,
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            results.append(result)
            print(
                f"completed {dataset_name}/{model_name}: "
                f"accuracy={result['accuracy']:.4f}, PR-AUC={result['pr_auc']:.4f}",
                file=sys.stderr,
            )
    print(json.dumps(results, indent=2))
    print_markdown(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_worker(args.dataset, args.model)))
    else:
        run_parent()


if __name__ == "__main__":
    main()
