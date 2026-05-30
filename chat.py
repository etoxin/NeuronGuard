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


def chat():
    print("======================================================================")
    print("🧠 Welcome to the NeuronGuard-Gen (v1.0-Alpha) Interactive Chat!")
    print("======================================================================")

    vocab_file = "wikipedia_vocab.txt"
    weights_file = "wikipedia_weights.txt"

    if not os.path.exists(vocab_file) or not os.path.exists(weights_file):
        print("⚠️ Pre-trained weights or vocabulary file not found!")
        print("💡 Please run the training pipeline first using:")
        print("   python train_generative_poc.py")
        return

    print("Loading vocabulary...")
    with open(vocab_file, "r") as f:
        vocab = json.load(f)

    tokenizer = NeuronGuardTokenizer(vocab_size=50000)
    tokenizer.vocab = vocab
    tokenizer.inverse_vocab = {v: k for k, v in vocab.items()}

    print("Allocating neuromorphic memory matrix...")
    trainer_field = ng.NeuronGuardTrainerField(sensory_count=50000, motor_count=50000)

    print("Loading pre-trained synaptic weights...")
    trainer_field.load_weights_from_b64(weights_file)
    print("Weights loaded successfully! SNN-LM is ready.")
    print("======================================================================")
    print("Type your prompt and press Enter to chat. Type 'exit' or 'quit' to stop.")
    print("======================================================================")

    while True:
        try:
            prompt = input("\n👤 You: ")
            if prompt.strip().lower() in ["exit", "quit"]:
                break

            if not prompt.strip():
                continue

            prompt_ids = tokenizer.encode(prompt)

            # Reset potentials for a clean generation session
            trainer_field.reset_potentials()

            # Process the seed prompt to establish initial CPG loopback energy
            trainer_field.process_step_sync(prompt_ids)

            generated_sequence = []
            current_token_id = prompt_ids[-1]

            # Autoregressive generation loop
            max_generation_length = 20
            temperature = 0.7

            for _ in range(max_generation_length):
                trainer_field.process_step_sync([current_token_id])

                # Pull raw potentials
                raw_potentials = np.array(
                    trainer_field.get_potentials(), dtype=np.float32
                )

                # Apply stochastic Temperature layer
                scaled_logits = raw_potentials / max(temperature, 1e-5)

                # Softmax
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                probabilities = exp_logits / exp_logits.sum()

                # Top-10 sampling pool filter
                top_indices = np.argpartition(probabilities, -10)[-10:]
                top_probs = probabilities[top_indices]
                top_probs /= top_probs.sum()

                sampled_token_id = int(np.random.choice(top_indices, p=top_probs))

                if sampled_token_id == 49999:  # EOS marker
                    break

                generated_sequence.append(sampled_token_id)
                current_token_id = sampled_token_id

                # Apply decay
                trainer_field.decay_potentials(0.90)

            response = tokenizer.decode(generated_sequence)
            print(f"🧠 NeuronGuard: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Error during generation: {e}")


if __name__ == "__main__":
    chat()
