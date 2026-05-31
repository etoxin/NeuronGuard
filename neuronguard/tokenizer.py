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
Unified tokenizer for NeuronGuard text classification.

Provides a single, consistent tokenization pipeline used by all classifiers.
Replaces the 3 different ad-hoc tokenizers previously scattered across examples.
"""

import re

# Comprehensive stop words list — union of all example lists plus common noise words
DEFAULT_STOP_WORDS = frozenset({
    # Determiners & articles
    "the", "a", "an", "this", "that", "these", "those",
    # Prepositions
    "of", "to", "in", "on", "for", "with", "at", "by", "from", "as",
    "into", "about", "between", "through", "over", "after", "before",
    # Conjunctions
    "and", "but", "or", "not", "nor",
    # Pronouns
    "it", "its", "he", "him", "his", "she", "her", "they", "their",
    "we", "you", "me", "who", "which", "what", "where", "when", "how",
    # Common verbs
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "will", "would", "could", "should", "may", "can",
    "did", "does", "do",
    # Adverbs & misc
    "also", "just", "more", "than", "other", "some", "such", "many", "most",
    "first", "last", "each", "made", "said", "new", "one", "two", "three",
})

# Pre-compiled regex for word boundary tokenization
_WORD_PATTERN = re.compile(r"\b[a-z0-9]+\b")

# Minimum token length to keep
MIN_TOKEN_LENGTH = 3


def stem(word):
    """Lightweight suffix-stripping stemmer.

    Collapses common English inflections without external dependencies.
    Intentionally conservative to avoid over-stemming.

    Args:
        word: A lowercase string token.

    Returns:
        The stemmed form of the word.
    """
    if len(word) <= 4:
        return word

    if word.endswith("tion"):
        return word[:-4]
    if word.endswith("sion"):
        return word[:-4]
    if word.endswith("ment"):
        return word[:-4]
    if word.endswith("ness"):
        return word[:-4]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    return word


def tokenize(text, stop_words=None, apply_stemming=True, min_length=MIN_TOKEN_LENGTH):
    """Tokenize text into a list of clean, normalized word tokens.

    Pipeline: lowercase → extract word boundaries → filter stop words →
    filter by length → optional stemming.

    Args:
        text: The input text string to tokenize.
        stop_words: Set of words to exclude. Defaults to DEFAULT_STOP_WORDS.
        apply_stemming: Whether to apply lightweight suffix stripping.
        min_length: Minimum token length to keep.

    Returns:
        A list of cleaned, normalized token strings.
    """
    if stop_words is None:
        stop_words = DEFAULT_STOP_WORDS

    # Lowercase and extract word-boundary tokens
    tokens = _WORD_PATTERN.findall(text.lower())

    # Filter stop words and short tokens
    tokens = [t for t in tokens if len(t) >= min_length and t not in stop_words]

    # Apply lightweight stemming
    if apply_stemming:
        tokens = [stem(t) for t in tokens]

    return tokens
