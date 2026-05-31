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

import json
import os

import neuronguard as ng
import numpy as np
from neuronguard import NeuronGuardTokenizer


def setup_isolated_test_env():
    vocab_size = 50000
    temperature = 0.0001  # Absolute minimum to enforce greedy deterministic sampling

    # Define the hermetic text boundaries
    controlled_sentences = [
        "ducks are yellow and white .",
        "the lake is blue and cold .",
        "we walk down to the park .",
        "ducks swim on the lake .",
    ]

    # Build dynamic, un-collided, symmetric index maps
    unique_words = sorted(list(set(" ".join(controlled_sentences).split())))
    vocab = {word: idx for idx, word in enumerate(unique_words)}
    vocab["<EOS>"] = 49999

    vocab_file = "wikipedia_vocab.txt"
    with open(vocab_file, "w") as f:
        json.dump(vocab, f)

    # Initialize the trainer field with sensory and motor counts
    trainer_field = ng.NeuronGuardTrainerField(vocab_size, vocab_size)

    tokenizer = NeuronGuardTokenizer(vocab_size=vocab_size)
    tokenizer.vocab = vocab
    tokenizer.inverse_vocab = {v: k for k, v in vocab.items()}

    return trainer_field, tokenizer, controlled_sentences, temperature


def execute_integration_verification():
    trainer_field, tokenizer, sentences, temp = setup_isolated_test_env()

    print("⚡ Step 1: Potentiating Synapses Over Isolated Epochs...")
    import random

    random.seed(42)  # Fixed seed for 100% repeatable deterministic shuffling
    for epoch in range(20):
        shuffled_sentences = list(sentences)
        random.shuffle(shuffled_sentences)
        for sentence in shuffled_sentences:
            token_ids = [
                tokenizer.vocab[w] for w in sentence.split() if w in tokenizer.vocab
            ]
            trainer_field.train_stream_step_sync(token_ids)

    print("✅ System Potentiated. Launching Factual Trajectory Trace...")
    eval_prompt = "what color are ducks"

    # Prime TPI Registers
    prompt_ids = [
        tokenizer.vocab[w] for w in eval_prompt.split() if w in tokenizer.vocab
    ]
    for token_id in prompt_ids:
        trainer_field.process_step_sync([token_id])
        trainer_field.decay_potentials(0.95)  # 5% intra-phrase leakage

    print(f"\nPrompt Input: {eval_prompt}")
    print("Predicted Trace: ", end="")

    # Greedy Verification Loop
    for _ in range(5):
        raw_potentials = np.array(trainer_field.get_potentials(), dtype=np.float32)

        # Apply Contrast Exponentiation (PRD 6.0.0 compliance)
        max_p = np.max(raw_potentials)
        logits = (raw_potentials / max_p) ** 3 if max_p > 0 else raw_potentials

        sampled_id = int(np.argmax(logits))
        if sampled_id == 49999:
            break

        word = tokenizer.inverse_vocab.get(sampled_id, "[Unknown]")
        print(f"{word} ", end="", flush=True)

        # Advance state-machine clocks
        trainer_field.process_step_sync([sampled_id])
        trainer_field.decay_potentials(
            0.90
        )  # Reverted back to 0.90 as specified in the PRD
    print("\n")


if __name__ == "__main__":
    execute_integration_verification()
