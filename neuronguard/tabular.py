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
import gzip
import json
import math
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
        learning_rule: str = "log_likelihood",
        smoothing: float = 1.0,
        weight_scale: float = 256.0,
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
            bucket_strategy (str, optional): "uniform" or exact "quantile" buckets. Defaults to "uniform".
            decision_threshold (float, optional): Binary class-1 score-margin threshold. Defaults to 0.
            learning_rule (str, optional): Normalized "log_likelihood" or legacy "hebbian" updates. Defaults to "log_likelihood".
            smoothing (float, optional): Additive smoothing for likelihood weights. Defaults to 1.
            weight_scale (float, optional): Fixed-point log-weight scale. Defaults to 256.
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
        if learning_rule not in {"log_likelihood", "hebbian"}:
            raise ValueError("learning_rule must be 'log_likelihood' or 'hebbian'")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if weight_scale <= 0:
            raise ValueError("weight_scale must be positive")
        self.learning_rule = learning_rule
        self.smoothing = float(smoothing)
        self.weight_scale = float(weight_scale)

        self.num_sensory = num_features * buckets_per_feature
        if self.use_feature_interactions:
            self.num_sensory += self.interaction_vocab_size
        self._bias_token: Optional[int] = None
        if self.learning_rule == "log_likelihood":
            self._bias_token = self.num_sensory
            self.num_sensory += 1

        self._field: Optional[Any] = None
        self._features_min: Optional[List[float]] = None
        self._features_max: Optional[List[float]] = None
        self._bucket_boundaries: Optional[List[List[float]]] = None
        self._token_class_counts: Dict[int, List[float]] = {}
        self._class_counts: List[float] = [0.0] * self.num_classes
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

        if len(records) == 0:
            raise ValueError("cannot fit bucket boundaries from an empty dataset")

        if self.bucket_strategy == "quantile":
            self._features_min = []
            self._features_max = []
            self._bucket_boundaries = []
            # Process one feature at a time so peak memory is O(records), not
            # O(records × features), while retaining exact quantiles.
            for feature_index in feature_indices:
                ordered = sorted(float(record[feature_index]) for record in records)
                self._features_min.append(ordered[0])
                self._features_max.append(ordered[-1])
                boundaries = []
                for bucket_index in range(1, self.buckets_per_feature):
                    value_index = min(
                        len(ordered) - 1,
                        (bucket_index * len(ordered)) // self.buckets_per_feature,
                    )
                    boundaries.append(ordered[value_index])
                self._bucket_boundaries.append(boundaries)
        else:
            self._features_min = [float("inf")] * self.num_features
            self._features_max = [float("-inf")] * self.num_features
            for record in records:
                for index, feature_index in enumerate(feature_indices):
                    value = float(record[feature_index])
                    self._features_min[index] = min(
                        self._features_min[index], value
                    )
                    self._features_max[index] = max(
                        self._features_max[index], value
                    )
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

        if self._bias_token is not None:
            tokens.append(self._bias_token)

        return tokens

    def _weight_from_log_probability(self, probability: float) -> int:
        weight = round(math.log(probability) * self.weight_scale)
        return max(-32768, min(32767, weight))

    def _likelihood_denominator(self, class_index: int, token: int) -> float:
        class_count = self._class_counts[class_index]
        base_token_count = self.num_features * self.buckets_per_feature
        if token < base_token_count:
            outcomes = self.buckets_per_feature
            activations_per_record = 1
        else:
            outcomes = self.interaction_vocab_size
            activations_per_record = self.num_features * (self.num_features - 1) // 2
        return (
            class_count * activations_per_record + self.smoothing * outcomes
        )

    def _refresh_likelihood_weights(self) -> None:
        if self.learning_rule != "log_likelihood":
            return

        base_token_count = self.num_features * self.buckets_per_feature
        tokens_to_refresh = set(range(base_token_count))
        tokens_to_refresh.update(
            token
            for token in self._token_class_counts
            if token != self._bias_token
        )

        empty_counts = [0.0] * self.num_classes
        for token in tokens_to_refresh:
            counts = self._token_class_counts.get(token, empty_counts)
            synapses = []
            for class_index in range(self.num_classes):
                probability = (counts[class_index] + self.smoothing) / (
                    self._likelihood_denominator(class_index, token)
                )
                synapses.append(
                    (class_index, self._weight_from_log_probability(probability))
                )
            self._field.replace_neuron_synapses(token, synapses)

        total_count = sum(self._class_counts)
        prior_denominator = total_count + self.smoothing * self.num_classes
        prior_synapses = []
        for class_index in range(self.num_classes):
            prior = (self._class_counts[class_index] + self.smoothing) / prior_denominator
            prior_synapses.append(
                (class_index, self._weight_from_log_probability(prior))
            )
        self._field.replace_neuron_synapses(self._bias_token, prior_synapses)

    def _accumulate_likelihood(
        self, label: int, tokens: Iterable[int], weight: float
    ) -> None:
        self._class_counts[label] += weight
        for token in tokens:
            if token == self._bias_token:
                continue
            counts = self._token_class_counts.setdefault(
                token, [0.0] * self.num_classes
            )
            counts[label] += weight

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

        # Initialise a fresh field so repeated fit() calls replace prior state.
        self._field = None
        self._ensure_field()
        if self.learning_rule == "hebbian" and self.baseline_delta:
            self._seed_baseline(default_class)

        if epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rule == "log_likelihood":
            self._token_class_counts = {}
            self._class_counts = [0.0] * self.num_classes
            for record in records:
                try:
                    label = int(record[label_index])
                    features = [record[index] for index in feature_indices]
                except (TypeError, ValueError, IndexError):
                    continue
                if not 0 <= label < self.num_classes:
                    raise ValueError(f"label {label} is outside the configured classes")
                weight = class_weights.get(label, 1)
                if weight <= 0:
                    raise ValueError("class weights must be positive")
                self._accumulate_likelihood(
                    label,
                    self._get_tokens(features),
                    float(weight),
                )
            self._refresh_likelihood_weights()
        else:
            train_records = []
            for record in records:
                try:
                    label = int(record[label_index])
                    features = [record[index] for index in feature_indices]
                except (TypeError, ValueError, IndexError):
                    continue
                if not 0 <= label < self.num_classes:
                    raise ValueError(f"label {label} is outside the configured classes")
                weight = class_weights.get(label, 1)
                if weight <= 0:
                    raise ValueError("class weights must be positive")
                train_records.append((label, self._get_tokens(features), weight))
            for _ in range(epochs):
                if shuffle:
                    random.shuffle(train_records)
                for label, tokens, weight in train_records:
                    for _ in range(weight):
                        self._field.train_stream(
                            tokens, label, self.amplify_delta, self.suppress_delta
                        )

        self._is_fitted = True

    def fit_xy(
        self,
        features,
        labels,
        epochs: int = 1,
        shuffle: bool = True,
        class_weights: Optional[Dict[int, int]] = None,
        default_class: int = 0,
    ) -> None:
        """Fit directly from separate feature and label arrays.

        This avoids constructing combined Python records and works with NumPy
        arrays without making NumPy a runtime dependency.
        """
        labels = list(labels)
        if len(features) != len(labels):
            raise ValueError("features and labels must contain the same number of rows")
        if not labels:
            raise ValueError("cannot fit an empty dataset")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        class_weights = class_weights or {}
        feature_indices = list(range(self.num_features))
        self._compute_boundaries(features, feature_indices)

        self._field = None
        self._ensure_field()
        if self.learning_rule == "hebbian" and self.baseline_delta:
            self._seed_baseline(default_class)

        if self.learning_rule == "log_likelihood":
            self._token_class_counts = {}
            self._class_counts = [0.0] * self.num_classes
            for row, raw_label in zip(features, labels):
                label = int(raw_label)
                if not 0 <= label < self.num_classes:
                    raise ValueError(f"label {label} is outside the configured classes")
                weight = class_weights.get(label, 1)
                if weight <= 0:
                    raise ValueError("class weights must be positive")
                self._accumulate_likelihood(
                    label, self._get_tokens(row), float(weight)
                )
            self._refresh_likelihood_weights()
        else:
            train_records = []
            for row, raw_label in zip(features, labels):
                label = int(raw_label)
                if not 0 <= label < self.num_classes:
                    raise ValueError(f"label {label} is outside the configured classes")
                weight = class_weights.get(label, 1)
                train_records.append((label, self._get_tokens(row), weight))
            for _ in range(epochs):
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

        if epochs < 1:
            raise ValueError("epochs must be positive")
        self._field = None
        self._ensure_field()
        if self.learning_rule == "hebbian" and self.baseline_delta:
            self._seed_baseline(default_class)
        else:
            self._token_class_counts = {}
            self._class_counts = [0.0] * self.num_classes

        # Pass 2 to N: Training
        training_passes = 1 if self.learning_rule == "log_likelihood" else epochs
        for epoch in range(training_passes):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delimiter)
                if skip_header:
                    next(reader, None)
                for row in reader:
                    try:
                        label = int(float(row[label_index]))
                        features = [row[fi] for fi in feature_indices]
                        tokens = self._get_tokens(features)
                        if not 0 <= label < self.num_classes:
                            raise ValueError(
                                f"label {label} is outside the configured classes"
                            )
                        weight = class_weights.get(label, 1)
                        if weight <= 0:
                            raise ValueError("class weights must be positive")
                        if self.learning_rule == "log_likelihood":
                            self._accumulate_likelihood(label, tokens, float(weight))
                        else:
                            for _ in range(weight):
                                self._field.train_stream(
                                    tokens,
                                    label,
                                    self.amplify_delta,
                                    self.suppress_delta,
                                )
                    except (ValueError, IndexError):
                        continue

        if self.learning_rule == "log_likelihood":
            self._refresh_likelihood_weights()

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
            
        records = list(X)
        if not records:
            return
        class_weights = class_weights or {}
        feature_indices = [i for i in range(len(records[0])) if i != label_index]
        if len(feature_indices) != self.num_features:
            raise ValueError(f"expected {self.num_features} features")

        for record in records:
            label = int(record[label_index])
            if not 0 <= label < self.num_classes:
                raise ValueError(f"label {label} is outside the configured classes")
            features = [record[i] for i in feature_indices]
            indices = self._get_tokens(features)

            weight = class_weights.get(label, 1)
            if weight <= 0:
                raise ValueError("class weights must be positive")
            if self.learning_rule == "log_likelihood":
                self._accumulate_likelihood(label, indices, float(weight))
            elif indices:
                self._field.train_stream(
                    indices,
                    label,
                    self.amplify_delta * weight,
                    self.suppress_delta * weight,
                )

        if self.learning_rule == "log_likelihood":
            # Every class denominator changes after an update, so refresh every
            # learned token rather than leaving untouched weights stale.
            self._refresh_likelihood_weights()

    def unlearn(
        self,
        X: Iterable[Union[List[float], Tuple[float, ...]]],
        label_index: int,
        class_weights: Optional[Dict[int, int]] = None,
    ) -> None:
        """Remove record counts or apply inverse legacy Hebbian deltas.
        
        Args:
            X (Iterable[Union[List[float], Tuple[float, ...]]]): Iterable of lists of floats (features) with the label appended.
            label_index (int): The index of the label in each record.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before it can be unlearned.")
            
        records = list(X)
        if not records:
            return
        class_weights = class_weights or {}
        feature_indices = [i for i in range(len(records[0])) if i != label_index]
        if len(feature_indices) != self.num_features:
            raise ValueError(f"expected {self.num_features} features")

        for record in records:
            label = int(record[label_index])
            features = [record[i] for i in feature_indices]
            indices = self._get_tokens(features)
            weight = class_weights.get(label, 1)
            if weight <= 0:
                raise ValueError("class weights must be positive")

            if self.learning_rule == "log_likelihood":
                if self._class_counts[label] < weight:
                    raise ValueError("cannot unlearn more records than were learned")
                self._class_counts[label] -= weight
                for token in indices:
                    if token == self._bias_token:
                        continue
                    counts = self._token_class_counts.get(token)
                    if counts is None or counts[label] < weight:
                        raise ValueError("cannot unlearn an unknown token association")
                    counts[label] -= weight
            elif indices:
                self._field.train_stream(
                    indices,
                    label,
                    -self.amplify_delta,
                    -self.suppress_delta,
                )

        if self.learning_rule == "log_likelihood":
            self._refresh_likelihood_weights()

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

    def predict_proba(self, features: List[float]) -> List[float]:
        """Return normalized class scores using a stable softmax transform."""
        scores = self.predict_scores(features)
        scale = self.weight_scale if self.learning_rule == "log_likelihood" else 1.0
        scaled = [score / scale for score in scores]
        maximum = max(scaled)
        exponentials = [math.exp(score - maximum) for score in scaled]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def predict_scores_batch(self, features, batch_size: int = 4096) -> List[List[int]]:
        """Return score vectors for many rows using native parallel batches."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        results = []
        for start in range(0, len(features), batch_size):
            rows = features[start : start + batch_size]
            token_rows = []
            for row in rows:
                if len(row) != self.num_features:
                    raise ValueError(
                        f"expected {self.num_features} features, got {len(row)}"
                    )
                token_rows.append(self._get_tokens(row))
            results.extend(self._field.predict_scores_batch(token_rows))
        return results

    def predict_batch(self, features, batch_size: int = 4096) -> List[int]:
        """Predict many rows while retaining tuned binary-threshold semantics."""
        score_rows = self.predict_scores_batch(features, batch_size=batch_size)
        if self.num_classes == 2:
            return [
                1 if scores[1] - scores[0] >= self.decision_threshold else 0
                for scores in score_rows
            ]
        return [scores.index(max(scores)) for scores in score_rows]

    def predict_proba_batch(
        self, features, batch_size: int = 4096
    ) -> List[List[float]]:
        """Return normalized class scores for many feature rows."""
        scale = self.weight_scale if self.learning_rule == "log_likelihood" else 1.0
        probabilities = []
        for scores in self.predict_scores_batch(features, batch_size=batch_size):
            scaled = [score / scale for score in scores]
            maximum = max(scaled)
            exponentials = [math.exp(score - maximum) for score in scaled]
            total = sum(exponentials)
            probabilities.append([value / total for value in exponentials])
        return probabilities

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
        for record in records:
            label = int(record[label_index])
            if label not in (0, 1):
                raise ValueError("binary threshold labels must be 0 or 1")
            features = [record[index] for index in feature_indices]
            ranked.append((self.predict_margin(features), label))

        return self._select_decision_threshold(ranked)

    def tune_decision_threshold_xy(self, features, labels) -> float:
        """Tune the binary F1 threshold from separate validation arrays."""
        labels = list(labels)
        if len(features) != len(labels):
            raise ValueError("features and labels must contain the same number of rows")
        ranked = []
        for row, raw_label in zip(features, labels):
            label = int(raw_label)
            if label not in (0, 1):
                raise ValueError("binary threshold labels must be 0 or 1")
            ranked.append((self.predict_margin(row), label))
        return self._select_decision_threshold(ranked)

    def _select_decision_threshold(self, ranked) -> float:
        positive_count = sum(label for _, label in ranked)

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

        if self.learning_rule == "log_likelihood":
            count_state = {
                "token_class_counts": self._token_class_counts,
                "class_counts": self._class_counts,
            }
            with gzip.open(
                os.path.join(path, "counts.json.gz"), "wt", encoding="utf-8"
            ) as count_file:
                json.dump(count_state, count_file, separators=(",", ":"))

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
            "learning_rule": self.learning_rule,
            "smoothing": self.smoothing,
            "weight_scale": self.weight_scale,
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
            learning_rule=config.get("learning_rule", "hebbian"),
            smoothing=config.get("smoothing", 1.0),
            weight_scale=config.get("weight_scale", 256.0),
        )

        classifier._features_min = config["features_min"]
        classifier._features_max = config["features_max"]
        classifier._bucket_boundaries = config.get("bucket_boundaries")
        counts_path = os.path.join(path, "counts.json.gz")
        if os.path.exists(counts_path):
            with gzip.open(counts_path, "rt", encoding="utf-8") as count_file:
                count_state = json.load(count_file)
        else:
            count_state = config
        classifier._token_class_counts = {
            int(token): counts
            for token, counts in count_state.get("token_class_counts", {}).items()
        }
        classifier._class_counts = count_state.get(
            "class_counts", [0.0] * classifier.num_classes
        )

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
