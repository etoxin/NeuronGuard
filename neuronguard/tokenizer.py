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
from typing import Set, Optional, List, FrozenSet

# Comprehensive stop words list — union of all example lists plus common noise words
DEFAULT_STOP_WORDS: FrozenSet[str] = frozenset({
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

MIN_TOKEN_LENGTH: int = 3

from .neuronguard import tokenize as _rust_tokenize

def tokenize(
    text: str,
    stop_words: Optional[Set[str]] = None,
    apply_stemming: bool = True,
    min_length: int = MIN_TOKEN_LENGTH,
) -> List[str]:
    """Tokenize a string into a list of words.

    Args:
        text (str): The input text to tokenize.
        stop_words (Optional[Set[str]], optional): A set of stop words to exclude. Defaults to None, which uses DEFAULT_STOP_WORDS.
        apply_stemming (bool, optional): Whether to apply stemming to the tokens. Defaults to True.
        min_length (int, optional): Minimum length for a token to be kept. Defaults to MIN_TOKEN_LENGTH.

    Returns:
        List[str]: A list of processed token strings.

    Examples:
        >>> tokenize("The quick brown foxes!", apply_stemming=True)
        ['quick', 'brown', 'fox']
    """
    if stop_words is None:
        stop_words = set(DEFAULT_STOP_WORDS)
    return _rust_tokenize(text, stop_words, apply_stemming, min_length)
