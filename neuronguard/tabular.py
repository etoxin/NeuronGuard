# Copyright 2026 Adam Lusted
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TabularClassifier: High-level tabular/numerical classification API for NeuronGuard.

Wraps the bare-metal NeuronGuardField with automatic feature bucketing, class
weighting for imbalanced datasets, and atomic prediction.

Replaces the manual bucket-computation and oversampling boilerplate in the
fraud scanner example.
"""

import bisect
import json
import os
import random
from typing import Dict, List, Optional, Tuple, Any, Iterable, Union


class TabularClassifier:
    """High-level tabular data classifier backed by a NeuronGuardField.

    Automatically buckets continuous features into discrete sensory neuron
    indices and handles class imbalance through configurable oversampling.

    Example::

        classifier = TabularClassifier(num_classes=2, num_features=5)
        classifier.fit(
            records=train_data,
            feature_indices=[0, 1, 2, 3, 4],
            label_index=5,
            class_weights={1: 100},
        )
        label = classifier.predict([v10, v12, v14, v17, amount])
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        buckets_per_feature: int = 10,
        amplify_delta: int = 15,
        suppress_delta: int = 5,
        baseline_delta: int = 10,
        use_feature_interactions: bool = False,
        interaction_vocab_size: int = 1000000,
        bucket_strategy: str = "uniform",
        decision_threshold: float = 0.0,
    ) -> None:
        """Initialise a TabularClassifier.

        Args:
            num_classes (int): Number of output categories.
            num_features (int): Number of input features.
            buckets_per_feature (int, optional): Number of discrete buckets per feature. Defaults to 10.
            amplify_delta (int, optional): Weight increment for the correct class during training. Defaults to 15.
            suppress_delta (int, optional): Weight decrement for incorrect classes during training. Defaults to 5.
            baseline_delta (int, optional): Weight used for initial baseline seeding. Defaults to 10.
            use_feature_interactions (bool, optional): If True, hashes pairs of features to capture 2D non-linear patterns. Defaults to False.
            interaction_vocab_size (int, optional): Size of the hash space for interactions to prevent collisions. Defaults to 1000000.
        """
        if not 1 <= num_classes <= 8:
            raise ValueError("num_classes must be between 1 and 8")
        if num_features < 1:
            raise ValueError("num_features must be positive")
        if buckets_per_feature < 2:
            raise ValueError("buckets_per_feature must be at least 2")
        if use_feature_interactions and interaction_vocab_size < 1:
            raise ValueError("interaction_vocab_size must be positive")
        self.num_classes = num_classes
        self.num_features = num_features
        self.buckets_per_feature = buckets_per_feature
        self.amplify_delta = amplify_delta
        self.suppress_delta = suppress_delta
        self.baseline_delta = baseline_delta
        self.use_feature_interactions = use_feature_interactions
        self.interaction_vocab_size = interaction_vocab_size
        if bucket_strategy not in {"uniform", "quantile"}:
            raise ValueError("bucket_strategy must be 'uniform' or 'quantile'")
        self.bucket_strategy = bucket_strategy
        self.decision_threshold = float(decision_threshold)

        self.num_sensory = num_features * buckets_per_feature
        if self.use_feature_interactions:
            self.num_sensory += self.interaction_vocab_size

        self._field: Optional[Any] = None
        self._features_min: Optional[List[float]] = None
        self._features_max: Optional[List[float]] = None
        self._bucket_boundaries: Optional[List[List[float]]] = None
        self._is_fitted: bool = False

    def _ensure_field(self) -> None:
        """Lazily import and create the Rust NeuronGuardField."""
        if self._field is None:
            from .neuronguard import NeuronGuardField

            self._field = NeuronGuardField(
                sensory_count=self.num_sensory, motor_count=self.num_classes
            )

    def _compute_boundaries(self, records: Iterable[Union[List[float], Tuple[float, ...]]], feature_indices: List[int]) -> None:
        """Compute min/max boundaries for each feature from training data.
        
        Args:
            records (Iterable[Union[List[float], Tuple[float, ...]]]): Training records.
            feature_indices (List[int]): Indices of the features.
        """
        if len(feature_indices) != self.num_features:
            raise ValueError(
                f"expected {self.num_features} feature indices, got {len(feature_indices)}"
            )

        feature_values = [[] for _ in range(self.num_features)]

        for record in records:
            for i, fi in enumerate(feature_indices):
                val = float(record[fi])
                feature_values[i].append(val)

        if any(not values for values in feature_values):
            raise ValueError("cannot fit bucket boundaries from an empty dataset")

        self._features_min = [min(values) for values in feature_values]
        self._features_max = [max(values) for values in feature_values]
        if self.bucket_strategy == "quantile":
            self._bucket_boundaries = []
            for values in feature_values:
                ordered = sorted(values)
                boundaries = []
                for bucket_index in range(1, self.buckets_per_feature):
                    value_index = min(
                        len(ordered) - 1,
                        (bucket_index * len(ordered)) // self.buckets_per_feature,
                    )
                    boundaries.append(ordered[value_index])
                self._bucket_boundaries.append(boundaries)
        else:
            self._bucket_boundaries = None

    def _get_tokens(self, features: List[float]) -> List[int]:
        """Convert a list of feature values to sensory neuron indices.

        Args:
            features (List[float]): List of numerical feature values (same order as feature_indices).

        Returns:
            List[int]: List of sensory neuron indices.
        """
        tokens = []
        for i in range(self.num_features):
            val = float(features[i])
            min_val = self._features_min[i]
            max_val = self._features_max[i]

            if self.bucket_strategy == "quantile":
                bucket = bisect.bisect_right(self._bucket_boundaries[i], val)
                bucket = min(bucket, self.buckets_per_feature - 1)
            elif max_val > min_val:
                if val <= min_val:
                    bucket = 0
                elif val >= max_val:
                    bucket = self.buckets_per_feature - 1
                else:
                    bucket = int(
                        (val - min_val) / (max_val - min_val) * self.buckets_per_feature
                    )
                    bucket = min(bucket, self.buckets_per_feature - 1)
            else:
                bucket = 0

            tokens.append(i * self.buckets_per_feature + bucket)
            
        if self.use_feature_interactions:
            interaction_offset = self.num_features * self.buckets_per_feature
            num_base_tokens = len(tokens)
            for i in range(num_base_tokens):
                for j in range(i + 1, num_base_tokens):
                    # Deterministic fast hash for a pair of integers
                    pair_hash = (tokens[i] * 83492791 + tokens[j]) % self.interaction_vocab_size
                    tokens.append(interaction_offset + pair_hash)
                    
        return tokens

    def _seed_baseline(self, default_class: int = 0) -> None:
        """Seed all sensory neurons to a default class (e.g., legitimate).
        
        Args:
            default_class (int, optional): The class to seed to. Defaults to 0.
        """
        for i in range(self.num_sensory):
            self._field.train_stream([i], default_class, self.baseline_delta, 0)

    # -------------------------------------------------------------------------
    # Fitting
    # -------------------------------------------------------------------------

    def fit(
        self,
        records: Iterable[Union[List[float], Tuple[float, ...]]],
        feature_indices: List[int],
        label_index: int,
        epochs: int = 1,
        shuffle: bool = True,
        class_weights: Optional[Dict[int, int]] = None,
        default_class: int = 0,
    ) -> None:
        """Train the classifier on tabular records.

        Args:
            records (Iterable[Union[List[float], Tuple[float, ...]]]): List of records (lists/tuples of values).
            feature_indices (List[int]): List of column indices for input features.
            label_index (int): Column index for the integer class label.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle records before each epoch. Defaults to True.
            class_weights (Optional[Dict[int, int]], optional): Optional dict mapping class_label → oversample_multiplier.
                For example, {1: 100} trains fraud cases 100 times per epoch. Defaults to None.
            default_class (int, optional): The class to seed all neurons to initially (e.g., 0 for "legitimate"). Defaults to 0.
        """
        records = list(records)
        if class_weights is None:
            class_weights = {}

        # Compute feature boundaries
        self._compute_boundaries(records, feature_indices)

        # Initialise field and seed baseline
        self._ensure_field()
        self._seed_baseline(default_class)

        # Build tokenized training records
        train_records = []
        for record in records:
            try:
                label = int(record[label_index])
                features = [record[fi] for fi in feature_indices]
                tokens = self._get_tokens(features)
                weight = class_weights.get(label, 1)
                train_records.append((label, tokens, weight))
            except (ValueError, IndexError):
                continue

        # Multi-epoch training
        for epoch in range(epochs):
            if shuffle:
                random.shuffle(train_records)
            for label, tokens, weight in train_records:
                for _ in range(weight):
                    self._field.train_stream(
                        tokens, label, self.amplify_delta, self.suppress_delta
                    )

        self._is_fitted = True

    def fit_from_csv(
        self,
        file_path: str,
        feature_indices: List[int],
        label_index: int,
        epochs: int = 1,
        class_weights: Optional[Dict[int, int]] = None,
        default_class: int = 0,
        delimiter: str = ",",
        skip_header: bool = False
    ) -> None:
        """Train the classifier by streaming directly from a CSV file.
        
        This uses O(1) memory and is designed for massive datasets (10M+ rows)
        that cannot fit in RAM. It makes multiple passes over the file.

        Args:
            file_path (str): Path to the CSV file.
            feature_indices (List[int]): List of column indices for input features.
            label_index (int): Column index for the integer class label.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            class_weights (Optional[Dict[int, int]], optional): Optional dict mapping class_label -> oversample_multiplier. Defaults to None.
            default_class (int, optional): The class to seed all neurons to initially. Defaults to 0.
            delimiter (str, optional): CSV delimiter. Defaults to ",".
            skip_header (bool, optional): Whether to skip the first row. Defaults to False.
        """
        import csv
        if self.bucket_strategy == "quantile":
            raise ValueError(
                "exact quantile bucketing requires fit() with in-memory records"
            )
        if class_weights is None:
            class_weights = {}

        # Pass 1: Compute Boundaries
        self._features_min = [float("inf")] * self.num_features
        self._features_max = [float("-inf")] * self.num_features
        
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            if skip_header:
                next(reader, None)
            for row in reader:
                for i, fi in enumerate(feature_indices):
                    try:
                        val = float(row[fi])
                        if val < self._features_min[i]:
                            self._features_min[i] = val
                        if val > self._features_max[i]:
                            self._features_max[i] = val
                    except (ValueError, IndexError):
                        continue

        self._ensure_field()
        self._seed_baseline(default_class)

        # Pass 2 to N: Training
        for epoch in range(epochs):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delimiter)
                if skip_header:
                    next(reader, None)
                for row in reader:
                    try:
                        label = int(float(row[label_index]))
                        features = [row[fi] for fi in feature_indices]
                        tokens = self._get_tokens(features)
                        weight = class_weights.get(label, 1)
                        for _ in range(weight):
                            self._field.train_stream(
                                tokens, label, self.amplify_delta, self.suppress_delta
                            )
                    except (ValueError, IndexError):
                        continue

        self._is_fitted = True

    def update(self, X: Iterable[Union[List[float], Tuple[float, ...]]], label_index: int, class_weights: Optional[Dict[int, int]] = None) -> None:
        """Continually learn from new records on the fly.
        
        This updates the existing model weights without rebuilding bucket boundaries.
        
        Args:
            X (Iterable[Union[List[float], Tuple[float, ...]]]): Iterable of lists of floats (features) with the label appended.
            label_index (int): The index of the label in each record.
            class_weights (Optional[Dict[int, int]], optional): Optional dict mapping class_label -> oversample_multiplier. Defaults to None.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before it can be updated.")
            
        if class_weights is None:
            class_weights = {}
            
        feature_indices = [i for i in range(len(X[0])) if i != label_index]
        
        for record in X:
            label = int(record[label_index])
            features = [record[i] for i in feature_indices]
            indices = self._get_tokens(features)
                    
            if indices:
                weight = class_weights.get(label, 1)
                self._field.train_stream(
                    indices, 
                    label, 
                    self.amplify_delta * weight, 
                    self.suppress_delta * weight
                )

    def unlearn(self, X: Iterable[Union[List[float], Tuple[float, ...]]], label_index: int) -> None:
        """Surgically unlearn records by applying negative Hebbian deltas.
        
        Args:
            X (Iterable[Union[List[float], Tuple[float, ...]]]): Iterable of lists of floats (features) with the label appended.
            label_index (int): The index of the label in each record.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before it can be unlearned.")
            
        feature_indices = [i for i in range(len(X[0])) if i != label_index]
        
        for record in X:
            label = int(record[label_index])
            features = [record[i] for i in feature_indices]
            indices = self._get_tokens(features)
                    
            if indices:
                self._field.train_stream(
                    indices, 
                    label, 
                    -self.amplify_delta, 
                    -self.suppress_delta
                )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict(self, features: List[float]) -> int:
        """Classify a feature vector and return the predicted class index.

        Atomically handles reset → tokenize → process → argmax.

        Args:
            features (List[float]): List of numerical feature values (same order as training features).

        Returns:
            int: The predicted class index (0-indexed).
        """
        scores = self.predict_scores(features)
        if self.num_classes == 2:
            margin = scores[1] - scores[0]
            return 1 if margin >= self.decision_threshold else 0
        return scores.index(max(scores))

    def predict_scores(self, features: List[float]) -> List[int]:
        """Classify a feature vector and return raw potentials for all classes.

        Args:
            features (List[float]): List of numerical feature values.

        Returns:
            List[int]: A list of integer potentials, one per class.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction.")
        if len(features) != self.num_features:
            raise ValueError(
                f"expected {self.num_features} features, got {len(features)}"
            )
        return self._field.predict_scores(self._get_tokens(features))

    def predict_margin(self, features: List[float]) -> int:
        """Return the class-1 minus class-0 score for a binary classifier."""
        if self.num_classes != 2:
            raise ValueError("predict_margin is only available for binary classifiers")
        scores = self.predict_scores(features)
        return scores[1] - scores[0]

    def tune_decision_threshold(
        self,
        records: Iterable[Union[List[float], Tuple[float, ...]]],
        feature_indices: List[int],
        label_index: int,
    ) -> float:
        """Select the validation-set score threshold that maximizes binary F1.

        The supplied records must be held out from model training. The selected
        threshold is stored on the classifier and used by subsequent predictions.
        """
        if self.num_classes != 2:
            raise ValueError("decision-threshold tuning requires two classes")

        ranked = []
        positive_count = 0
        for record in records:
            label = int(record[label_index])
            if label not in (0, 1):
                raise ValueError("binary threshold labels must be 0 or 1")
            features = [record[index] for index in feature_indices]
            ranked.append((self.predict_margin(features), label))
            positive_count += label

        if not ranked or positive_count == 0 or positive_count == len(ranked):
            raise ValueError("threshold tuning requires records from both classes")

        ranked.sort(key=lambda item: item[0], reverse=True)
        true_positives = 0
        false_positives = 0
        best_f1 = -1.0
        best_threshold = float(ranked[0][0])
        index = 0
        while index < len(ranked):
            threshold = ranked[index][0]
            while index < len(ranked) and ranked[index][0] == threshold:
                if ranked[index][1] == 1:
                    true_positives += 1
                else:
                    false_positives += 1
                index += 1

            false_negatives = positive_count - true_positives
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        self.decision_threshold = best_threshold
        return best_threshold

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate(self, records: Iterable[Union[List[float], Tuple[float, ...]]], feature_indices: List[int], label_index: int) -> Tuple[float, str]:
        """Evaluate accuracy on test records.

        Args:
            records (Iterable[Union[List[float], Tuple[float, ...]]]): List of test records.
            feature_indices (List[int]): List of column indices for input features.
            label_index (int): Column index for the class label.

        Returns:
            Tuple[float, str]: A tuple of (accuracy_pct, report_str) with per-class metrics.
        """
        confusion = [[0] * self.num_classes for _ in range(self.num_classes)]
        correct = 0
        total = 0

        for record in records:
            try:
                actual = int(record[label_index])
                features = [record[fi] for fi in feature_indices]
            except (ValueError, IndexError):
                continue

            predicted = self.predict(features)
            confusion[actual][predicted] += 1
            if predicted == actual:
                correct += 1
            total += 1

        accuracy = (correct / total * 100) if total > 0 else 0.0
        report = self._format_report(confusion, correct, total, accuracy)
        return accuracy, report

    def evaluate_from_csv(self, file_path: str, feature_indices: List[int], label_index: int, delimiter: str = ",", skip_header: bool = False) -> Tuple[float, str]:
        """Evaluate accuracy by streaming directly from a CSV file.
        
        Args:
            file_path (str): Path to the test CSV file.
            feature_indices (List[int]): List of column indices for input features.
            label_index (int): Column index for the class label.
            delimiter (str, optional): CSV delimiter. Defaults to ",".
            skip_header (bool, optional): Whether to skip the first row. Defaults to False.

        Returns:
            Tuple[float, str]: A tuple of (accuracy_pct, report_str) with per-class metrics.
        """
        import csv
        confusion = [[0] * self.num_classes for _ in range(self.num_classes)]
        correct = 0
        total = 0

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            if skip_header:
                next(reader, None)
            for row in reader:
                try:
                    actual = int(float(row[label_index]))
                    features = [row[fi] for fi in feature_indices]
                except (ValueError, IndexError):
                    continue

                predicted = self.predict(features)
                confusion[actual][predicted] += 1
                if predicted == actual:
                    correct += 1
                total += 1

        accuracy = (correct / total * 100) if total > 0 else 0.0
        report = self._format_report(confusion, correct, total, accuracy)
        return accuracy, report

    def _format_report(self, confusion: List[List[int]], correct: int, total: int, accuracy: float) -> str:
        """Format a classification report with per-class metrics.
        
        Args:
            confusion (List[List[int]]): Confusion matrix.
            correct (int): Number of correctly predicted samples.
            total (int): Total number of samples.
            accuracy (float): Overall accuracy percentage.

        Returns:
            str: Formatted classification report.
        """
        lines = []
        lines.append(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
        lines.append("")
        lines.append(
            f"{'Class':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}"
        )
        lines.append("-" * 53)

        for i in range(self.num_classes):
            tp = confusion[i][i]
            fp = sum(confusion[j][i] for j in range(self.num_classes)) - tp
            fn = sum(confusion[i][j] for j in range(self.num_classes)) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            lines.append(
                f"Class {i:<9} | {precision * 100:9.2f}% | {recall * 100:9.2f}% | {f1 * 100:9.2f}%"
            )

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained model to a directory.

        Args:
            path (str): Directory path to save the model to.
        """
        os.makedirs(path, exist_ok=True)

        self._field.save_weights(os.path.join(path, "weights.bin"))

        config = {
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "buckets_per_feature": self.buckets_per_feature,
            "amplify_delta": self.amplify_delta,
            "suppress_delta": self.suppress_delta,
            "baseline_delta": self.baseline_delta,
            "features_min": self._features_min,
            "features_max": self._features_max,
            "bucket_strategy": self.bucket_strategy,
            "bucket_boundaries": self._bucket_boundaries,
            "decision_threshold": self.decision_threshold,
            "use_feature_interactions": self.use_feature_interactions,
            "interaction_vocab_size": self.interaction_vocab_size,
        }
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TabularClassifier":
        """Load a trained model from a directory.

        Args:
            path (str): Directory path containing weights.bin and config.json.

        Returns:
            TabularClassifier: A fitted TabularClassifier instance.
        """
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        classifier = cls(
            num_classes=config["num_classes"],
            num_features=config["num_features"],
            buckets_per_feature=config.get("buckets_per_feature", 10),
            amplify_delta=config.get("amplify_delta", 15),
            suppress_delta=config.get("suppress_delta", 5),
            baseline_delta=config.get("baseline_delta", 10),
            bucket_strategy=config.get("bucket_strategy", "uniform"),
            decision_threshold=config.get("decision_threshold", 0.0),
            use_feature_interactions=config.get("use_feature_interactions", False),
            interaction_vocab_size=config.get("interaction_vocab_size", 1000000),
        )

        classifier._features_min = config["features_min"]
        classifier._features_max = config["features_max"]
        classifier._bucket_boundaries = config.get("bucket_boundaries")

        classifier._ensure_field()
        classifier._field.load_weights(os.path.join(path, "weights.bin"))
        classifier._is_fitted = True

        return classifier

    @staticmethod
    def exists(path: str) -> bool:
        """Check whether a saved model exists at the given path.
        
        Args:
            path (str): Directory path to check.

        Returns:
            bool: True if weights and config exist.
        """
        return os.path.exists(os.path.join(path, "weights.bin")) and os.path.exists(
            os.path.join(path, "config.json")
        )
