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

import re


class NeuronGuardTokenizer:
    """
    NeuronGuardTokenizer
    A highly compressed, local Byte-Pair Encoding (BPE) dictionary mapping
    subword fragments down to flat u16 Token IDs.
    """

    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        # Initialize a basic vocabulary with ASCII characters and common subwords
        self.vocab = {chr(i): i for i in range(256)}
        self.inverse_vocab = {i: chr(i) for i in range(256)}

        # Add some common subwords to simulate a trained BPE vocabulary up to vocab_size
        common_subwords = [
            "the",
            "and",
            "ing",
            "ion",
            "ent",
            "for",
            "that",
            "tis",
            "es",
            "en",
            "to",
            "it",
            "is",
            "was",
            "he",
            "she",
            "his",
            "her",
            "in",
            "on",
            "at",
            "by",
            "an",
            "with",
            "this",
            "you",
            "not",
            "but",
            "or",
            "as",
        ]
        for i, subword in enumerate(common_subwords):
            token_id = 256 + i
            self.vocab[subword] = token_id
            self.inverse_vocab[token_id] = subword

    def encode(self, text):
        """
        Encodes a string into a list of Token IDs using greedy subword matching.
        """
        tokens = []
        i = 0
        while i < len(text):
            match = None
            # Try to find the longest matching subword in vocabulary
            for length in range(min(10, len(text) - i), 0, -1):
                subword = text[i : i + length]
                if subword in self.vocab:
                    match = subword
                    break
            if match:
                tokens.append(self.vocab[match])
                i += len(match)
            else:
                # Fallback to byte value
                tokens.append(ord(text[i]) % 256)
                i += 1
        return tokens

    def decode(self, token_ids):
        """
        Decodes a list of Token IDs back into a string.
        """
        return "".join(self.inverse_vocab.get(tid, "?") for tid in token_ids)

    def split_topological_fields(self, text):
        """
        Splits input text concurrently across three semantic fields:
        - Field 0: Lexical word fragments (Token IDs).
        - Field 1: Formatting syntax and structure density.
        - Field 2: Micro-temporal distance profiles between adjacent nouns/entities.
        """
        # Field 0: Lexical word fragments
        field_0 = self.encode(text)

        # Field 1: Formatting syntax and structure density
        # We can quantize features like punctuation, capitalization, and spacing
        field_1 = []
        for char in text:
            if char.isupper():
                field_1.append(1)  # Capitalization
            elif char in ".,!?;:()[]{}":
                field_1.append(2)  # Punctuation
            elif char.isspace():
                field_1.append(3)  # Space
            else:
                field_1.append(0)  # Lexical/Other
        # Pad or truncate Field 1 to match Field 0 length
        if len(field_1) < len(field_0):
            field_1 += [0] * (len(field_0) - len(field_1))
        else:
            field_1 = field_1[: len(field_0)]

        # Field 2: Micro-temporal distance profiles between adjacent nouns/entities
        # We can identify capitalized words or common nouns and compute distances between them
        words = re.findall(r"\w+|[^\w\s]", text)
        noun_indices = []
        for idx, word in enumerate(words):
            # Simple heuristic for nouns/entities: capitalized words or words with length > 4
            if word and (word[0].isupper() or len(word) > 4):
                noun_indices.append(idx)

        # Map distances to tokens
        field_2 = []
        for idx in range(len(words)):
            # Find the distance to the next noun
            next_nouns = [n_idx for n_idx in noun_indices if n_idx > idx]
            if next_nouns:
                field_2.append(min(next_nouns[0] - idx, 255))  # Cap at 255
            else:
                field_2.append(0)

        # Pad or truncate Field 2 to match Field 0 length
        if len(field_2) < len(field_0):
            field_2 += [0] * (len(field_0) - len(field_2))
        else:
            field_2 = field_2[: len(field_0)]

        return field_0, field_1, field_2
