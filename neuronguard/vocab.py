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
Discriminative vocabulary builder for NeuronGuard text classification.

Replaces the raw-frequency vocabulary selection used in previous examples
with a scoring function that prioritises words concentrated in fewer categories.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Any
from .tokenizer import tokenize


def discriminative_score(counts: List[int]) -> float:
    """Score a word by frequency × category concentration.

    A word appearing 1000 times with 80% in one category scores 800.
    A word appearing 2000 times uniformly across 4 categories scores 500.
    The discriminative word wins the vocabulary slot.

    Args:
        counts (List[int]): List of per-category occurrence counts.

    Returns:
        float: A float score (higher = more discriminative).

    Examples:
        >>> discriminative_score([800, 100, 100])
        800.0
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    max_ratio = max(counts) / total
    return total * max_ratio


def frequency_score(counts: List[int]) -> float:
    """Score a word by raw total frequency (legacy behaviour).

    Args:
        counts (List[int]): List of per-category occurrence counts.

    Returns:
        float: Total frequency as a float.
    """
    return float(sum(counts))


def build_vocab(
    records: Iterable[Tuple[int, str]],
    num_classes: int,
    vocab_size: int,
    stop_words: Optional[Set[str]] = None,
    scoring: str = "discriminative",
    apply_stemming: bool = True,
) -> Tuple[Dict[str, int], List[Tuple[str, List[int], int]], Dict[str, Any]]:
    """Build a vocabulary from training records using discriminative scoring.

    Args:
        records (Iterable[Tuple[int, str]]): Iterable of (label, text) tuples where label is an int
            (0-indexed class) and text is the raw input string.
        num_classes (int): Number of output classes.
        vocab_size (int): Maximum vocabulary size.
        stop_words (Optional[Set[str]], optional): Optional set of stop words (defaults to tokenizer's built-in set). Defaults to None.
        scoring (str, optional): Scoring strategy — "discriminative" or "frequency". Defaults to "discriminative".
        apply_stemming (bool, optional): Whether to apply lightweight stemming during tokenization. Defaults to True.

    Returns:
        Tuple[Dict[str, int], List[Tuple[str, List[int], int]], Dict[str, Any]]: A tuple of (vocab_map, vocab_list, vocab_stats) where:
            - vocab_map: dict mapping word → index (0-indexed)
            - vocab_list: list of (word, per_class_counts, total_count) tuples
            - vocab_stats: dict of summary statistics

    Examples:
        >>> build_vocab([(0, "hello world")], 2, 1000)
        ({'hello': 0, 'world': 1}, [('hello', [1, 0], 1), ('world', [1, 0], 1)], {'total_records': 1, 'unique_words_seen': 2, 'vocab_size': 2, 'scoring': 'discriminative'})
    """
    score_fn = discriminative_score if scoring == "discriminative" else frequency_score

    # Count per-class word frequencies
    word_counts = {}
    total_records = 0

    for label, text in records:
        tokens = tokenize(text, stop_words=stop_words, apply_stemming=apply_stemming)
        for token in tokens:
            if token not in word_counts:
                word_counts[token] = [0] * num_classes
            word_counts[token][label] += 1
        total_records += 1

    # Score and sort
    word_list = []
    for word, counts in word_counts.items():
        total_count = sum(counts)
        word_list.append((word, counts, total_count))

    word_list.sort(key=lambda x: score_fn(x[1]), reverse=True)
    final_vocab = word_list[:vocab_size]

    # Build map
    vocab_map = {}
    for idx, (word, _, _) in enumerate(final_vocab):
        vocab_map[word] = idx

    vocab_stats = {
        "total_records": total_records,
        "unique_words_seen": len(word_counts),
        "vocab_size": len(final_vocab),
        "scoring": scoring,
    }

    return vocab_map, final_vocab, vocab_stats
