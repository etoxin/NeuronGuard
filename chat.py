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


def chat():
    print("======================================================================")
    print("🧠 Welcome to the NeuronGuard-Gen (v1.0-Alpha) Interactive Chat!")
    print("======================================================================")

    vocab_size = int(os.environ.get("VOCAB_SIZE", 50000))
    temperature = float(os.environ.get("TEMPERATURE", 0.6))

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

            # Reset potentials for a clean generation session
            trainer_field.reset_potentials()

            # Process seed prompt to establish context
            trainer_field.process_step_sync(prompt_ids)

            generated_sequence = []
            current_token_id = prompt_ids[-1]

            max_generation_length = 30
            print("🧠 NeuronGuard: ", end="", flush=True)

            for _ in range(max_generation_length):
                raw_potentials = np.array(
                    trainer_field.get_potentials(), dtype=np.float32
                )

                # 1. HARD REFRACTORY FILTER: Absolute suppression of recently fired ghosts
                for token_id in set(generated_sequence[-15:]):
                    raw_potentials[token_id] = -9999.0

                # 2. LOGIT NORMALIZATION: Scale down massive integer steps before exponential math
                max_potential = np.max(raw_potentials)
                if max_potential > 0:
                    normalized_logits = raw_potentials / max_potential
                else:
                    normalized_logits = raw_potentials

                # Apply Temperature over the normalized field
                scaled_logits = normalized_logits / max(temperature, 1e-5)

                # Stable Softmax execution
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                probabilities = exp_logits / exp_logits.sum()

                # Top-10 sampling pool filter
                top_indices = np.argpartition(probabilities, -10)[-10:]
                top_probs = probabilities[top_indices]

                sum_top_probs = top_probs.sum()
                if sum_top_probs > 0:
                    top_probs /= sum_top_probs
                else:
                    top_probs = np.ones_like(top_probs) / len(top_probs)

                sampled_token_id = int(np.random.choice(top_indices, p=top_probs))

                if sampled_token_id == 49999:  # Check EOS boundary
                    break

                generated_sequence.append(sampled_token_id)
                current_token_id = sampled_token_id

                # Advance the event step and decay state balances by 10%
                trainer_field.process_step_sync([current_token_id])
                trainer_field.decay_potentials(0.90)

            # Reconstruct and print final sentence layout
            response = tokenizer.decode(generated_sequence)
            print(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠️ Error during generation: {e}")


if __name__ == "__main__":
    chat()
