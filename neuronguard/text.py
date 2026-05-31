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
from typing import Dict, List, Optional, Set, Tuple, Any, Iterable, Union


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
        num_classes: int,
        vocab_size: int = 5000,
        amplify_delta: int = 15,
        suppress_delta: int = 5,
        seed_max_weight: int = 30,
        stop_words: Optional[Iterable[str]] = None,
        apply_stemming: bool = True,
        vocab_scoring: str = "discriminative",
        class_names: Optional[List[str]] = None,
        use_hashed_bigrams: bool = False,
    ) -> None:
        """Initialise a TextClassifier.

        Args:
            num_classes (int): Number of output categories.
            vocab_size (int, optional): Maximum vocabulary size (number of sensory neurons). Defaults to 5000.
            amplify_delta (int, optional): Weight increment for the correct class during training. Defaults to 15.
            suppress_delta (int, optional): Weight decrement for incorrect classes during training. Defaults to 5.
            seed_max_weight (int, optional): Maximum weight used when pre-seeding category distributions. Defaults to 30.
            stop_words (Optional[Iterable[str]], optional): Optional custom stop words set. Defaults to built-in comprehensive list.
            apply_stemming (bool, optional): Whether to apply lightweight suffix stripping. Defaults to True.
            vocab_scoring (str, optional): Vocabulary scoring strategy — "discriminative" or "frequency". Defaults to "discriminative".
            class_names (Optional[List[str]], optional): Optional list of human-readable class names. Defaults to None.
            use_hashed_bigrams (bool, optional): Whether to use hashed bigrams. Defaults to False.
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
        self.use_hashed_bigrams = use_hashed_bigrams

        # These are populated during fit() or load()
        self._field: Optional[Any] = None
        self._vocab_map: Dict[str, Union[int, List[int]]] = {}
        self._vocab_list: List[Tuple[str, List[int], int]] = []
        self._is_fitted: bool = False

    def _ensure_field(self) -> None:
        """Lazily import and create the Rust NeuronGuardField."""
        if self._field is None:
            from .neuronguard import NeuronGuardField

            self._field = NeuronGuardField(
                sensory_count=self.vocab_size, motor_count=self.num_classes
            )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using the classifier's configured pipeline.
        
        Args:
            text (str): Input text to tokenize.
            
        Returns:
            List[str]: List of token strings.
        """
        return tokenize(
            text,
            stop_words=self.stop_words,
            apply_stemming=self.apply_stemming,
        )

    def _text_to_indices(self, text: str) -> List[int]:
        """Convert text to a list of vocabulary indices.
        
        Args:
            text (str): Input text.
            
        Returns:
            List[int]: List of vocabulary indices.
        """
        tokens = self._tokenize(text)
        
        # Unigram indices (lower half of space if bigrams enabled)
        unigram_space = self.vocab_size // 2 if self.use_hashed_bigrams else self.vocab_size
        indices = []
        for t in tokens:
            if t in self._vocab_map:
                val = self._vocab_map[t]
                if isinstance(val, list):
                    indices.extend([v for v in val if v < unigram_space])
                elif val < unigram_space:
                    indices.append(val)
        
        # Hashed bigram indices (upper half of space)
        if self.use_hashed_bigrams and len(tokens) > 1:
            import zlib
            bigram_space = self.vocab_size - unigram_space
            for i in range(len(tokens) - 1):
                pair = f"{tokens[i]}_{tokens[i+1]}"
                bigram_id = (zlib.crc32(pair.encode('utf-8')) % bigram_space) + unigram_space
                indices.append(bigram_id)
                
        return indices

    def _seed_weights(self) -> None:
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

    def fit(self, train_file: str, text_col: Union[int, List[int]], label_col: int, epochs: int = 3, shuffle: bool = True, label_offset: int = -1) -> None:
        """Train the classifier from a CSV file.

        Args:
            train_file (str): Path to the training CSV file.
            text_col (Union[int, List[int]]): Column index (int) or list of column indices to concatenate as text.
            label_col (int): Column index containing the integer class label.
            epochs (int, optional): Number of training epochs with per-epoch shuffling. Defaults to 3.
            shuffle (bool, optional): Whether to shuffle records before each epoch. Defaults to True.
            label_offset (int, optional): Value subtracted from the raw label to get a 0-indexed class.
                Use -1 for 1-indexed CSV labels (default), 0 if labels are already 0-indexed. Defaults to -1.
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

        unigram_space = self.vocab_size // 2 if self.use_hashed_bigrams else self.vocab_size
        self._vocab_map, self._vocab_list, stats = build_vocab(
            records_for_vocab,
            self.num_classes,
            unigram_space,
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

    def fit_records(self, records: Iterable[Tuple[int, str]], epochs: int = 3, shuffle: bool = True) -> None:
        """Train the classifier from pre-parsed records.

        Args:
            records (Iterable[Tuple[int, str]]): Iterable of (label, text) tuples where label is a
                0-indexed int and text is the raw input string.
            epochs (int, optional): Number of training epochs. Defaults to 3.
            shuffle (bool, optional): Whether to shuffle records before each epoch. Defaults to True.
        """
        records = list(records)

        # Build vocabulary
        unigram_space = self.vocab_size // 2 if self.use_hashed_bigrams else self.vocab_size
        self._vocab_map, self._vocab_list, stats = build_vocab(
            records,
            self.num_classes,
            unigram_space,
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

    def update_records(self, records: Iterable[Tuple[int, str]]) -> None:
        """Continually learn from new records on the fly without rebuilding the vocabulary.
        
        This enables zero-overhead online/continuous learning. The model weights are
        updated instantly. Words not in the original vocabulary are ignored.
        
        Args:
            records (Iterable[Tuple[int, str]]): Iterable of (label, text) tuples.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before it can be updated.")
            
        for label, text in records:
            indices = self._text_to_indices(text)
            if indices:
                self._field.train_stream(
                    indices, label, self.amplify_delta, self.suppress_delta
                )

    def unlearn_records(self, records: Iterable[Tuple[int, str]]) -> None:
        """Instantly 'unlearn' records to comply with data privacy or correct errors.
        
        Because NeuronGuard uses reversible Hebbian plasticity rather than entangled
        gradient descent, you can cleanly subtract the exact synaptic weight modifications
        caused by a specific record. This solves the 'Machine Unlearning' problem instantly.
        
        Args:
            records (Iterable[Tuple[int, str]]): Iterable of (label, text) tuples to unlearn.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before it can be updated.")
            
        for label, text in records:
            indices = self._text_to_indices(text)
            if indices:
                # Invert the deltas to subtract the exact influence this record had
                self._field.train_stream(
                    indices, label, -self.amplify_delta, -self.suppress_delta
                )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict(self, text: str) -> int:
        """Classify text and return the predicted class index.

        Atomically handles reset → tokenize → process → argmax.
        Forgetting to reset potentials between predictions was a common
        silent accuracy bug in the raw API — this method makes it impossible.

        Args:
            text (str): The input text string to classify.

        Returns:
            int: The predicted class index (0-indexed).
        """
        indices = self._text_to_indices(text)
        self._field.reset_potentials()
        if indices:
            self._field.predict(indices)
        potentials = self._field.get_potentials()
        return potentials.index(max(potentials))

    def predict_scores(self, text: str) -> List[int]:
        """Classify text and return raw motor neuron potentials for all classes.

        Args:
            text (str): The input text string to classify.

        Returns:
            List[int]: A list of integer potentials, one per class.
        """
        indices = self._text_to_indices(text)
        self._field.reset_potentials()
        if indices:
            self._field.predict(indices)
        return self._field.get_potentials()

    def predict_name(self, text: str) -> str:
        """Classify text and return the human-readable class name.

        Requires class_names to be set during construction.

        Args:
            text (str): The input text string to classify.

        Returns:
            str: The predicted class name string.
        """
        idx = self.predict(text)
        if self.class_names and idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)

    # -------------------------------------------------------------------------
    # Diagnostics & Explainability
    # -------------------------------------------------------------------------

    def explain(self, text: str) -> Dict[str, Any]:
        """Provide a transparent, token-by-token explanation for a prediction.

        Because NeuronGuard is a direct associative memory rather than a black-box
        neural network, we can perfectly trace exactly which words contributed
        to the final prediction, and by exactly how much weight.

        Args:
            text (str): The input text string to classify.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'prediction': The predicted class index.
                - 'prediction_name': The predicted class name.
                - 'total_scores': Raw potentials for each class.
                - 'word_contributions': A list of dicts detailing each word's exact weight contribution.
        """
        self._ensure_field()
        tokens = self._tokenize(text)
        explanation = []
        for word in tokens:
            if word in self._vocab_map:
                val = self._vocab_map[word]
                contribs = {}
                
                # Handle SDRs
                if isinstance(val, list):
                    for idx in val:
                        synapses = self._field.get_neuron_synapses(idx)
                        for motor_id, weight in synapses:
                            if weight > 0:
                                label_str = self.class_names[motor_id] if self.class_names and motor_id < len(self.class_names) else str(motor_id)
                                contribs[label_str] = contribs.get(label_str, 0) + weight
                else:
                    synapses = self._field.get_neuron_synapses(val)
                    for motor_id, weight in synapses:
                        if weight > 0:
                            label_str = self.class_names[motor_id] if self.class_names and motor_id < len(self.class_names) else str(motor_id)
                            contribs[label_str] = weight
                            
                if contribs:
                    explanation.append({
                        "word": word,
                        "contributions": contribs
                    })
        
        prediction_idx = self.predict(text)
        prediction_name = self.predict_name(text)
        scores = self.predict_scores(text)
        
        return {
            "prediction": prediction_idx,
            "prediction_name": prediction_name,
            "total_scores": scores,
            "word_contributions": explanation
        }

    def get_class_features(self, class_idx: int, top_k: int = 10) -> List[Tuple[str, int]]:
        """Introspect the memory to find the most strongly associated words for a class.

        Args:
            class_idx (int): The class index to inspect.
            top_k (int, optional): Number of top words to return. Defaults to 10.

        Returns:
            List[Tuple[str, int]]: A list of (word, weight) tuples.
        """
        self._ensure_field()
        if not self._is_fitted:
            return []
            
        class_id = class_idx
        word_scores = []
        for word, val in self._vocab_map.items():
            score = 0
            # Handle SDRs (lists of indices)
            if isinstance(val, list):
                for idx in val:
                    synapses = self._field.get_neuron_synapses(idx)
                    for target_motor, weight in synapses:
                        if target_motor == class_id:
                            score += weight
            else:
                synapses = self._field.get_neuron_synapses(val)
                for target_motor, weight in synapses:
                    if target_motor == class_id:
                        score += weight
                        
            if score > 0:
                word_scores.append((word, score))
                    
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:top_k]
        
    def print_explanation(self, text: str) -> None:
        """Out-of-the-box diagnostic print for prediction explanations.
        
        Args:
            text (str): The input text string to classify.
        """
        explanation = self.explain(text)
        print(f"\n[Diagnostics] Input text: '{text}'")
        print(f"[Diagnostics] Final Prediction: {explanation['prediction_name']}")
        print(f"[Diagnostics] Raw Class Scores: {explanation['total_scores']}\n")
        
        if not explanation['word_contributions']:
            print("  (No vocabulary words recognized in text)")
            return
            
        print("Token Breakdown:")
        for contrib in explanation['word_contributions']:
            word = contrib['word']
            weights = contrib['contributions']
            weight_str = ", ".join([f"{cls}: {w:+}w" for cls, w in weights.items()])
            print(f"  '{word:<10}' -> {weight_str}")
            
    def print_class_features(self, class_idx: int, top_k: int = 10) -> None:
        """Out-of-the-box diagnostic print for class features.
        
        Args:
            class_idx (int): The class index to inspect.
            top_k (int, optional): Number of top words to return. Defaults to 10.
        """
        features = self.get_class_features(class_idx, top_k)
        class_name = self.class_names[class_idx] if self.class_names and class_idx < len(self.class_names) else str(class_idx)
        print(f"\n[Diagnostics] Top {top_k} memory triggers for class '{class_name}':")
        for word, weight in features:
            print(f"  - '{word}' (Weight: +{weight})")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate(self, test_file: str, text_col: Union[int, List[int]], label_col: int, label_offset: int = -1) -> Tuple[float, str]:
        """Evaluate accuracy on a test CSV file.

        Args:
            test_file (str): Path to the test CSV file.
            text_col (Union[int, List[int]]): Column index or list of indices for text.
            label_col (int): Column index for the integer class label.
            label_offset (int, optional): Value subtracted from raw label. Defaults to -1 for 1-indexed.

        Returns:
            Tuple[float, str]: A tuple of (accuracy_pct, report_str) where report_str is a
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

    def evaluate_records(self, records: Iterable[Tuple[int, str]]) -> Tuple[float, str]:
        """Evaluate accuracy on pre-parsed records.

        Args:
            records (Iterable[Tuple[int, str]]): Iterable of (label, text) tuples.

        Returns:
            Tuple[float, str]: A tuple of (accuracy_pct, report_str).
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

    def save(self, path: str) -> None:
        """Save the trained model to a directory.

        Creates a self-contained directory with weights, vocabulary, and config.
        Unlike the raw API (which saves weights and vocab separately), this
        bundles everything so they can never get out of sync.

        Args:
            path (str): Directory path to save the model to.
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
            "use_hashed_bigrams": getattr(self, "use_hashed_bigrams", False),
        }
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TextClassifier":
        """Load a trained model from a directory.

        Args:
            path (str): Directory path containing weights.bin, vocab.txt, and config.json.

        Returns:
            TextClassifier: A fitted TextClassifier instance.
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
            use_hashed_bigrams=config.get("use_hashed_bigrams", False),
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
    def exists(path: str) -> bool:
        """Check whether a saved model exists at the given path.

        Args:
            path (str): Directory path to check.

        Returns:
            bool: True if all required model files exist.
        """
        return (
            os.path.exists(os.path.join(path, "weights.bin"))
            and os.path.exists(os.path.join(path, "vocab.txt"))
            and os.path.exists(os.path.join(path, "config.json"))
        )
