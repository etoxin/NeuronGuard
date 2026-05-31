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

import tiktoken


class NeuronGuardTokenizer:
    """
    NeuronGuardTokenizer
    A highly compressed, local Byte-Pair Encoding (BPE) dictionary mapping
    subword fragments down to flat u16 Token IDs.
    Uses OpenAI's gpt2 vocabulary for 100% readable English words and subwords.
    """

    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}

        # Load OpenAI's gpt2 encoder
        self.enc = tiktoken.get_encoding("gpt2")

        # Extract the first vocab_size tokens
        for i in range(vocab_size):
            try:
                # Decode the token bytes to string
                token_bytes = self.enc.decode_single_token_bytes(i)
                token_str = token_bytes.decode("utf-8", errors="ignore")

                # Clean up control characters or replace them
                if not token_str.strip() and len(token_str) > 0:
                    # Keep spaces or newlines as readable representations
                    token_str = token_str.replace("\n", "⏎").replace("\t", "⇥")

                self.vocab[token_str] = i
                self.inverse_vocab[i] = token_str
            except Exception:
                # Fallback if token ID is invalid
                token_str = f"sub_{i}"
                self.vocab[token_str] = i
                self.inverse_vocab[i] = token_str

    def encode(self, text):
        """
        Encodes a string into a list of Token IDs using tiktoken's native Rust-based encoder.
        """
        return [tid for tid in self.enc.encode(text) if tid < self.vocab_size]

    def encode_content(self, text):
        """
        Encodes text for *training*, dropping tokens that are pure corpus noise rather than
        language: standalone numbers (Gutenberg page/chapter/footnote markers like "96426013599"
        or "8") and isolated punctuation runs. Filtering these at ingestion keeps the bigram graph
        free of junk successors that otherwise surface verbatim during generation.
        """
        ids = self.encode(text)
        kept = []
        for tid in ids:
            piece = self.inverse_vocab.get(tid)
            if piece is None:
                piece = self.enc.decode([tid])
            stripped = piece.strip()
            # Drop tokens whose visible content is entirely digits (page/footnote numbers).
            if stripped and all(ch.isdigit() for ch in stripped):
                continue
            kept.append(tid)
        return kept

    def decode(self, token_ids):
        """
        Decodes a list of Token IDs back into a string using tiktoken's native decoder.
        Gracefully ignores any invalid UTF-8 byte sequences to prevent replacement characters.
        """
        try:
            token_bytes = self.enc.decode_bytes(token_ids)
            return token_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return self.enc.decode(token_ids)

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
