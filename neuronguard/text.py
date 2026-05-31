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
TextClassifier: High-level text classification API for NeuronGuard.

Wraps the bare-metal NeuronGuardField Rust core with unified tokenization,
discriminative vocabulary building, correct default hyperparameters, and
a predict() method that handles reset → process → argmax atomically.

Replaces the 100-200 lines of boilerplate previously duplicated in every
text classification example.
"""

import csv
import json
import os
import random

from .tokenizer import DEFAULT_STOP_WORDS, tokenize
from .vocab import build_vocab


class TextClassifier:
    """High-level text classifier backed by a NeuronGuardField.

    Handles vocabulary building, weight seeding, multi-epoch shuffled training,
    and atomic prediction in a single clean API. All accuracy-critical defaults
    (discriminative scoring, 3:1 amplify/suppress ratio, proportional seeding)
    are baked in.

    Example::

        classifier = TextClassifier(num_classes=4, vocab_size=1000)
        classifier.fit("train.csv", text_col=[1, 2], label_col=0, epochs=3)
        accuracy, report = classifier.evaluate("test.csv", text_col=[1, 2], label_col=0)
        label = classifier.predict("some text to classify")
    """

    def __init__(
        self,
        num_classes,
        vocab_size=5000,
        amplify_delta=15,
        suppress_delta=5,
        seed_max_weight=30,
        stop_words=None,
        apply_stemming=True,
        vocab_scoring="discriminative",
        class_names=None,
    ):
        """Initialise a TextClassifier.

        Args:
            num_classes: Number of output categories.
            vocab_size: Maximum vocabulary size (number of sensory neurons).
            amplify_delta: Weight increment for the correct class during training.
            suppress_delta: Weight decrement for incorrect classes during training.
            seed_max_weight: Maximum weight used when pre-seeding category distributions.
            stop_words: Optional custom stop words set. Defaults to built-in comprehensive list.
            apply_stemming: Whether to apply lightweight suffix stripping.
            vocab_scoring: Vocabulary scoring strategy — "discriminative" or "frequency".
            class_names: Optional list of human-readable class names.
        """
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.amplify_delta = amplify_delta
        self.suppress_delta = suppress_delta
        self.seed_max_weight = seed_max_weight
        self.stop_words = set(stop_words) if stop_words else set(DEFAULT_STOP_WORDS)
        self.apply_stemming = apply_stemming
        self.vocab_scoring = vocab_scoring
        self.class_names = class_names

        # These are populated during fit() or load()
        self._field = None
        self._vocab_map = {}
        self._vocab_list = []
        self._is_fitted = False

    def _ensure_field(self):
        """Lazily import and create the Rust NeuronGuardField."""
        if self._field is None:
            from .neuronguard import NeuronGuardField

            self._field = NeuronGuardField(
                sensory_count=self.vocab_size, motor_count=self.num_classes
            )

    def _tokenize(self, text):
        """Tokenize text using the classifier's configured pipeline."""
        return tokenize(
            text,
            stop_words=self.stop_words,
            apply_stemming=self.apply_stemming,
        )

    def _text_to_indices(self, text):
        """Convert text to a list of vocabulary indices."""
        tokens = self._tokenize(text)
        return [self._vocab_map[t] for t in tokens if t in self._vocab_map]

    def _seed_weights(self):
        """Pre-seed neuron weights proportional to category distributions.

        Instead of assigning each word to a single dominant category with a
        flat weight, this seeds connections proportional to how concentrated
        each word is across categories.
        """
        for i, (word, counts, total) in enumerate(self._vocab_list):
            if i >= self.vocab_size or total == 0:
                break
            for cat_idx, count in enumerate(counts):
                if count > 0:
                    weight = int((count / total) * self.seed_max_weight)
                    if weight > 0:
                        self._field.train_stream([i], cat_idx, weight, 0)

    # -------------------------------------------------------------------------
    # Fitting
    # -------------------------------------------------------------------------

    def fit(self, train_file, text_col, label_col, epochs=3, shuffle=True, label_offset=-1):
        """Train the classifier from a CSV file.

        Args:
            train_file: Path to the training CSV file.
            text_col: Column index (int) or list of column indices to concatenate as text.
            label_col: Column index containing the integer class label.
            epochs: Number of training epochs with per-epoch shuffling.
            shuffle: Whether to shuffle records before each epoch.
            label_offset: Value subtracted from the raw label to get a 0-indexed class.
                Use -1 for 1-indexed CSV labels (default), 0 if labels are already 0-indexed.
        """
        if isinstance(text_col, int):
            text_col = [text_col]

        # --- Phase 1: Build vocabulary ---
        records_for_vocab = []
        with open(train_file, mode="r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for record in rdr:
                try:
                    label = int(record[label_col]) + label_offset
                    text = " ".join(record[c] for c in text_col)
                    records_for_vocab.append((label, text))
                except (ValueError, IndexError):
                    continue

        self._vocab_map, self._vocab_list, stats = build_vocab(
            records_for_vocab,
            self.num_classes,
            self.vocab_size,
            stop_words=self.stop_words,
            scoring=self.vocab_scoring,
            apply_stemming=self.apply_stemming,
        )

        # --- Phase 2: Initialise field and seed weights ---
        self._ensure_field()
        self._seed_weights()

        # --- Phase 3: Build tokenized training records ---
        train_records = []
        for label, text in records_for_vocab:
            indices = self._text_to_indices(text)
            if indices:
                train_records.append((label, indices))

        # --- Phase 4: Multi-epoch shuffled training ---
        for epoch in range(epochs):
            if shuffle:
                random.shuffle(train_records)
            for label, indices in train_records:
                self._field.train_stream(
                    indices, label, self.amplify_delta, self.suppress_delta
                )

        self._is_fitted = True

    def fit_records(self, records, epochs=3, shuffle=True):
        """Train the classifier from pre-parsed records.

        Args:
            records: Iterable of (label, text) tuples where label is a
                0-indexed int and text is the raw input string.
            epochs: Number of training epochs.
            shuffle: Whether to shuffle records before each epoch.
        """
        records = list(records)

        # Build vocabulary
        self._vocab_map, self._vocab_list, stats = build_vocab(
            records,
            self.num_classes,
            self.vocab_size,
            stop_words=self.stop_words,
            scoring=self.vocab_scoring,
            apply_stemming=self.apply_stemming,
        )

        # Initialise field and seed
        self._ensure_field()
        self._seed_weights()

        # Tokenize once
        train_records = []
        for label, text in records:
            indices = self._text_to_indices(text)
            if indices:
                train_records.append((label, indices))

        # Multi-epoch training
        for epoch in range(epochs):
            if shuffle:
                random.shuffle(train_records)
            for label, indices in train_records:
                self._field.train_stream(
                    indices, label, self.amplify_delta, self.suppress_delta
                )

        self._is_fitted = True

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict(self, text):
        """Classify text and return the predicted class index.

        Atomically handles reset → tokenize → process → argmax.
        Forgetting to reset potentials between predictions was a common
        silent accuracy bug in the raw API — this method makes it impossible.

        Args:
            text: The input text string to classify.

        Returns:
            The predicted class index (0-indexed).
        """
        indices = self._text_to_indices(text)
        self._field.reset_potentials()
        if indices:
            self._field.process_stream_sync(indices)
        potentials = self._field.get_potentials()
        return potentials.index(max(potentials))

    def predict_scores(self, text):
        """Classify text and return raw motor neuron potentials for all classes.

        Args:
            text: The input text string to classify.

        Returns:
            A list of integer potentials, one per class.
        """
        indices = self._text_to_indices(text)
        self._field.reset_potentials()
        if indices:
            self._field.process_stream_sync(indices)
        return self._field.get_potentials()

    def predict_name(self, text):
        """Classify text and return the human-readable class name.

        Requires class_names to be set during construction.

        Args:
            text: The input text string to classify.

        Returns:
            The predicted class name string.
        """
        idx = self.predict(text)
        if self.class_names and idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate(self, test_file, text_col, label_col, label_offset=-1):
        """Evaluate accuracy on a test CSV file.

        Args:
            test_file: Path to the test CSV file.
            text_col: Column index or list of indices for text.
            label_col: Column index for the integer class label.
            label_offset: Value subtracted from raw label (default -1 for 1-indexed).

        Returns:
            A tuple of (accuracy_pct, report_str) where report_str is a
            formatted table with per-class Precision, Recall, and F1.
        """
        if isinstance(text_col, int):
            text_col = [text_col]

        confusion = [[0] * self.num_classes for _ in range(self.num_classes)]
        correct = 0
        total = 0

        with open(test_file, mode="r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for record in rdr:
                try:
                    actual = int(record[label_col]) + label_offset
                    text = " ".join(record[c] for c in text_col)
                except (ValueError, IndexError):
                    continue

                predicted = self.predict(text)
                confusion[actual][predicted] += 1
                if predicted == actual:
                    correct += 1
                total += 1

        accuracy = (correct / total * 100) if total > 0 else 0.0
        report = self._format_report(confusion, correct, total, accuracy)
        return accuracy, report

    def evaluate_records(self, records):
        """Evaluate accuracy on pre-parsed records.

        Args:
            records: Iterable of (label, text) tuples.

        Returns:
            A tuple of (accuracy_pct, report_str).
        """
        confusion = [[0] * self.num_classes for _ in range(self.num_classes)]
        correct = 0
        total = 0

        for actual, text in records:
            predicted = self.predict(text)
            confusion[actual][predicted] += 1
            if predicted == actual:
                correct += 1
            total += 1

        accuracy = (correct / total * 100) if total > 0 else 0.0
        report = self._format_report(confusion, correct, total, accuracy)
        return accuracy, report

    def _format_report(self, confusion, correct, total, accuracy):
        """Format a classification report with per-class metrics."""
        lines = []
        lines.append(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
        lines.append("")
        lines.append(f"{'Category':<25} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
        lines.append("-" * 63)

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

            name = str(i)
            if self.class_names and i < len(self.class_names):
                name = self.class_names[i]
            if len(name) > 25:
                name = name[:22] + "..."

            lines.append(
                f"{name:<25} | {precision * 100:9.2f}% | {recall * 100:9.2f}% | {f1 * 100:9.2f}%"
            )

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path):
        """Save the trained model to a directory.

        Creates a self-contained directory with weights, vocabulary, and config.
        Unlike the raw API (which saves weights and vocab separately), this
        bundles everything so they can never get out of sync.

        Args:
            path: Directory path to save the model to.
        """
        os.makedirs(path, exist_ok=True)

        # Save weights
        self._field.save_weights(os.path.join(path, "weights.bin"))

        # Save vocabulary
        with open(os.path.join(path, "vocab.txt"), "w", encoding="utf-8") as f:
            for word, _, _ in self._vocab_list[: self.vocab_size]:
                f.write(f"{word}\n")

        # Save config
        config = {
            "num_classes": self.num_classes,
            "vocab_size": self.vocab_size,
            "amplify_delta": self.amplify_delta,
            "suppress_delta": self.suppress_delta,
            "seed_max_weight": self.seed_max_weight,
            "stop_words": sorted(self.stop_words),
            "apply_stemming": self.apply_stemming,
            "vocab_scoring": self.vocab_scoring,
            "class_names": self.class_names,
        }
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load a trained model from a directory.

        Args:
            path: Directory path containing weights.bin, vocab.txt, and config.json.

        Returns:
            A fitted TextClassifier instance.
        """
        # Load config
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        classifier = cls(
            num_classes=config["num_classes"],
            vocab_size=config["vocab_size"],
            amplify_delta=config.get("amplify_delta", 15),
            suppress_delta=config.get("suppress_delta", 5),
            seed_max_weight=config.get("seed_max_weight", 30),
            stop_words=set(config.get("stop_words", [])) or None,
            apply_stemming=config.get("apply_stemming", True),
            vocab_scoring=config.get("vocab_scoring", "discriminative"),
            class_names=config.get("class_names"),
        )

        # Load vocabulary
        classifier._vocab_list = []
        with open(os.path.join(path, "vocab.txt"), "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                word = line.strip()
                classifier._vocab_map[word] = idx
                classifier._vocab_list.append((word, [], 0))

        # Load weights
        classifier._ensure_field()
        classifier._field.load_weights(os.path.join(path, "weights.bin"))
        classifier._is_fitted = True

        return classifier

    @staticmethod
    def exists(path):
        """Check whether a saved model exists at the given path.

        Args:
            path: Directory path to check.

        Returns:
            True if all required model files exist.
        """
        return (
            os.path.exists(os.path.join(path, "weights.bin"))
            and os.path.exists(os.path.join(path, "vocab.txt"))
            and os.path.exists(os.path.join(path, "config.json"))
        )
