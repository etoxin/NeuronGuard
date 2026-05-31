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
    vocab_size = min(int(os.environ.get("VOCAB_SIZE", 50000)), MAX_ADDRESSABLE_NEURONS)
    temperature = float(os.environ.get("TEMPERATURE", 0.6))
    top_k = int(os.environ.get("TOP_K", 40))

    print("======================================================================")
    print(f"🧠 Scaling Network Allocation Layer: Horizontal Depth = {vocab_size} Rows")
    print("======================================================================")

    vocab_file = "wikipedia_vocab.txt"
    weights_file = "wikipedia_weights.txt"

    # Initialize or load an ultra-lean cache-optimized seed vocabulary
    if not os.path.exists(vocab_file):
        seed_words = ["hello", "what", "color", "is", "are", "the", "sky", "ducks", "yellow", "white", "blue", "."]
        vocab = {word: idx for idx, word in enumerate(seed_words)}
        vocab["<EOS>"] = 49999
        with open(vocab_file, "w") as f:
            json.dump(vocab, f)
    else:
        with open(vocab_file, "r") as f:
            vocab = json.load(f)

    inverse_vocab = {v: k for k, v in vocab.items()}

    print("Allocating Tabula Rasa Neuromorphic Matrix Field...")
    trainer_field = ng.NeuronGuardTrainerField(vocab_size, vocab_size)
    if os.path.exists(weights_file):
        trainer_field.load_weights_from_b64(weights_file)
        print("Weights loaded successfully! SNN-LM is ready.")

    tokenizer = NeuronGuardTokenizer(vocab_size=vocab_size)
    tokenizer.vocab = vocab
    tokenizer.inverse_vocab = inverse_vocab

    print("======================================================================")
    print("🧠 Live Interactive Learning Engine Active!")
    print("Commands: /correct <text> | /debug <word> | /spike <word> <mv> | /save | /reset | /vocab")
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

            # ─── INTERACTIVE SLASH-COMMAND ROUTER ─────────────────────────────────

            if user_input.startswith("/correct"):
                fact = user_input.replace("/correct", "").strip().lower()
                words = fact.split()

                # Dynamic token mapping for novel words
                for word in words:
                    if word not in tokenizer.vocab:
                        new_idx = len(tokenizer.vocab)
                        tokenizer.vocab[word] = new_idx
                        tokenizer.inverse_vocab[new_idx] = word

                # Synchronous real-time FFI path potentiation
                token_ids = [tokenizer.vocab[w] for w in words]
                for source, target in zip(token_ids[:-1], token_ids[1:]):
                    trainer_field.potentiate_synapse_sync(source, target, 50)
                print(f"✅ Synaptic paths reinforced (+50) for sequence.")
                continue

            elif user_input.startswith("/debug"):
                target_word = user_input.replace("/debug", "").strip().lower()
                if target_word in tokenizer.vocab:
                    target_idx = tokenizer.vocab[target_word]

                    # Fetch memory states straight from raw hardware lanes
                    current_voltage = trainer_field.get_potentials()[target_idx]
                    synapses = trainer_field.get_row_synapses_sync(target_idx)

                    print(f"\n🔬 [NEUROMORPHIC TELEMETRY: '{target_word}']")
                    print(f"📡 Row Address: {target_idx} | Node Potential: {current_voltage} mV")
                    print("─" * 60)
                    for slot, (tgt_id, weight) in enumerate(synapses):
                        if weight > 0:
                            tgt_word = tokenizer.inverse_vocab.get(tgt_id, f"ID:{tgt_id}")
                            print(f"  Slot [{slot:02d}] ──► {tgt_word:<12} (Weight: {weight:<4})")
                    print("─" * 60)
                else:
                    print(f"❌ '{target_word}' not found in current lexicon map.")
                continue

            elif user_input.startswith("/spike"):
                parts = user_input.split()
                if len(parts) == 3:
                    try:
                        word, voltage = parts[1].lower(), int(parts[2])
                        if word in tokenizer.vocab:
                            trainer_field.inject_potential_sync(tokenizer.vocab[word], voltage)
                            print(f"⚡ Injected {voltage}mV into node [{word}].")
                        else:
                            print(f"❌ '{word}' not found in current lexicon map.")
                    except ValueError:
                        print("❌ Invalid voltage value. Must be an integer.")
                else:
                    print("❌ Usage: /spike <word> <voltage>")
                continue

            elif user_input == "/save":
                trainer_field.save_weights_to_b64(weights_file)
                with open(vocab_file, "w") as f:
                    json.dump(tokenizer.vocab, f)
                print("💾 Matrix configuration cards frozen to local disk.")
                continue

            elif user_input == "/reset":
                trainer_field = ng.NeuronGuardTrainerField(vocab_size, vocab_size)
                # Overwrite/zero the vocab file too or keep the seed words?
                # The reset command resets the memory matrix to zero, but keeping tokenizer vocab is helpful.
                print("🔄 Memory field zeroed. System returned to Tabula Rasa.")
                continue

            elif user_input == "/vocab":
                print(f"📊 Dictionary Capacity: {len(tokenizer.vocab)} / {vocab_size} Rows")
                continue

            # ─── STANDARD INFERENCE GENERATION ────────────────────────────────────
            print("🧠 NeuronGuard: ", end="", flush=True)

            if 49999 in tokenizer.inverse_vocab:
                # Custom tabula rasa vocabulary (word-level)
                cleaned_input = user_input.lower().replace(".", " . ").replace("?", " ? ").replace("!", " ! ")
                prompt_ids = [tokenizer.vocab[w] for w in cleaned_input.split() if w in tokenizer.vocab]
            else:
                # GPT-2 subword vocabulary
                prompt_ids = tokenizer.encode(user_input)

            if not prompt_ids:
                print()
                continue

            generated_sequence = []
            current_token_id = prompt_ids[-1]

            # Light recency penalty
            recent_window = 4
            max_generation_length = 30

            for _ in range(max_generation_length):
                # Sample directly from the current token's LEARNED successor distribution
                sampled_token_id = trainer_field.sample_next_token(
                    current_token_id,
                    temperature,
                    top_k,
                    float(np.random.random()),
                )

                # No learned successor: stop gracefully.
                if sampled_token_id is None:
                    break

                # Avoid trivial immediate loops
                if sampled_token_id in generated_sequence[-recent_window:]:
                    retry = trainer_field.sample_next_token(
                        current_token_id,
                        temperature,
                        top_k,
                        float(np.random.random()),
                    )
                    if retry is not None:
                        sampled_token_id = retry

                if sampled_token_id in (49999, 50256):  # <EOS> or GPT-2 boundary
                    break

                generated_sequence.append(sampled_token_id)
                current_token_id = sampled_token_id

            # Reconstruct and print final sentence layout
            if 49999 in tokenizer.inverse_vocab:
                response = " ".join(tokenizer.inverse_vocab.get(tid, f"ID:{tid}") for tid in generated_sequence)
            else:
                response = tokenizer.decode(generated_sequence)
            print(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠️ Error during generation: {e}")



if __name__ == "__main__":
    chat()
