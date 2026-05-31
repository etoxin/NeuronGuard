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

    prompt_ids = [
        tokenizer.vocab[w] for w in eval_prompt.split() if w in tokenizer.vocab
    ]

    print(f"\nPrompt Input: {eval_prompt}")
    print("Predicted Trace: ", end="")

    # Greedy verification loop: with temperature ~0 and a uniform draw of 0.0, sampling
    # deterministically returns each token's single strongest learned successor. This walks the
    # variable-connection bigram graph directly instead of the old global potential field.
    current_token_id = prompt_ids[-1]
    generated = []
    for _ in range(5):
        sampled_id = trainer_field.sample_next_token(current_token_id, temp, 1, 0.0)
        if sampled_id is None or sampled_id == 49999:
            break

        word = tokenizer.inverse_vocab.get(sampled_id, "[Unknown]")
        print(f"{word} ", end="", flush=True)
        generated.append(word)
        current_token_id = sampled_id
    print("\n")

    # The corpus only ever follows "ducks" with "are" or "swim". A faithful bigram model must
    # continue with one of those, proving learned associations drive generation.
    assert generated, "model produced no continuation for 'ducks'"
    assert generated[0] in {"are", "swim"}, (
        f"expected 'ducks' to be followed by a learned successor (are/swim), got {generated[0]!r}"
    )
    print("✅ Semantic trajectory verification PASSED.")


if __name__ == "__main__":
    execute_integration_verification()
