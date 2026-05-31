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
import sys

import neuronguard as ng
import numpy as np
from neuronguard import NeuronGuardTokenizer

# Synaptic targets are stored as u16, so the field can address at most 65,536 neurons.
# Requesting more only allocates unreachable dead memory, so we clamp here to match the Rust core.
MAX_ADDRESSABLE_NEURONS = 65536


def chat():
    vocab_size = min(int(os.environ.get("VOCAB_SIZE", 50257)), MAX_ADDRESSABLE_NEURONS)
    temperature = float(os.environ.get("TEMPERATURE", 0.6))
    top_k = int(os.environ.get("TOP_K", 40))

    print("======================================================================")
    print(f"🧠 Scaling Network Allocation Layer: Horizontal Depth = {vocab_size} Rows")
    print("======================================================================")

    vocab_file = "wikipedia_vocab.txt"
    weights_file = "wikipedia_weights.txt"

    if not os.path.exists(vocab_file) or not os.path.exists(weights_file):
        print("⚠️ Pre-trained weights or vocabulary file not found!")
        print("💡 Please run the training pipeline first using:")
        print("    python train_stream_harvester.py")
        return

    print("Loading vocabulary...")
    with open(vocab_file, "r") as f:
        vocab = json.load(f)

    tokenizer = NeuronGuardTokenizer(vocab_size=vocab_size)
    tokenizer.vocab = vocab
    tokenizer.inverse_vocab = {v: k for k, v in vocab.items()}

    print("Allocating neuromorphic memory matrix...")
    trainer_field = ng.NeuronGuardTrainerField(
        sensory_count=vocab_size, motor_count=vocab_size
    )

    print("Loading pre-trained synaptic weights...")
    trainer_field.load_weights_from_b64(weights_file)
    print("Weights loaded successfully! SNN-LM is ready.")
    print("======================================================================")
    print("🧠 Live Feedback Mode Active!")
    print("Type normal prompts to converse. Use '/correct <text>' to guide the brain.")
    print("Type 'exit' or 'quit' to stop.")
    print("======================================================================")

    while True:
        try:
            prompt = input("\n👤 You: ")
            user_input = prompt.strip()
            if user_input.lower() in ["exit", "quit"]:
                break

            if not user_input:
                continue

            # Intercept real-time human correction signals
            if user_input.startswith("/correct "):
                target_text = user_input.replace("/correct ", "")
                target_indices = tokenizer.encode(target_text)

                if target_indices:
                    print(
                        "⚡ Live Hebbian Adjustment: Mutating active cache registers..."
                    )
                    trainer_field.train_stream_step_sync(target_indices)
                    print("✅ Synaptic paths modified. Test the prompt string again.")
                continue

            prompt_ids = tokenizer.encode(user_input)
            if not prompt_ids:
                continue

            generated_sequence = []
            current_token_id = prompt_ids[-1]

            # Light recency penalty: discourage immediate verbatim repetition without the old
            # hard 15-token ban that forced the model off its learned paths.
            recent_window = 4

            max_generation_length = 30
            print("🧠 NeuronGuard: ", end="", flush=True)

            for _ in range(max_generation_length):
                # Sample directly from the current token's LEARNED successor distribution
                # (cache-aligned core + variable-length overflow), weighted by synapse strength.
                sampled_token_id = trainer_field.sample_next_token(
                    current_token_id,
                    temperature,
                    top_k,
                    float(np.random.random()),
                )

                # No learned successor (dead-end token): stop gracefully.
                if sampled_token_id is None:
                    break

                # Avoid trivial immediate loops (a -> b -> a -> b ...): if we just emitted this
                # token very recently, take one more independent draw before giving up.
                if sampled_token_id in generated_sequence[-recent_window:]:
                    retry = trainer_field.sample_next_token(
                        current_token_id,
                        temperature,
                        top_k,
                        float(np.random.random()),
                    )
                    if retry is not None:
                        sampled_token_id = retry

                if sampled_token_id == 50256:  # GPT-2 <|endoftext|> boundary
                    break

                generated_sequence.append(sampled_token_id)
                current_token_id = sampled_token_id

            # Reconstruct and print final sentence layout
            response = tokenizer.decode(generated_sequence)
            print(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠️ Error during generation: {e}")


if __name__ == "__main__":
    chat()
