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

import neuronguard as ng
import numpy as np

END_OF_SEQUENCE_MARKER = 50000 - 1


def run_autoregressive_generation(
    prompt_token_ids, max_generation_length=100, temperature=0.7
):
    """
    Runs autoregressive generation using an isolated, session-specific context runner
    to prevent corrupting the model's read-only base weights.
    """
    # Initialize an isolated execution context inside the Rust layer
    session_field = ng.NeuronGuardField(sensory_count=50000, motor_count=50000)
    session_field.reset_potentials()

    # Process the seed prompt to establish initial CPG loopback energy
    session_field.process_stream_sync(prompt_token_ids)

    generated_sequence = []
    current_token_id = prompt_token_ids[-1]

    for _ in range(max_generation_length):
        # Step a single clock cycle pass
        session_field.process_stream_sync([current_token_id])

        # Pull raw integer potential balances directly via pointer view
        raw_potentials = np.array(session_field.get_potentials(), dtype=np.float32)

        # Apply stochastic Temperature layer to introduce lexical variance
        scaled_logits = raw_potentials / max(temperature, 1e-5)

        # Compute softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / exp_logits.sum()

        # Top-10 sampling pool filter to isolate coherent next-tokens
        top_indices = np.argpartition(probabilities, -10)[-10:]
        top_probs = probabilities[top_indices]
        top_probs /= top_probs.sum()

        sampled_token_id = int(np.random.choice(top_indices, p=top_probs))

        if sampled_token_id == END_OF_SEQUENCE_MARKER:
            break

        generated_sequence.append(sampled_token_id)
        current_token_id = sampled_token_id  # Seed back into sensory array

    return generated_sequence
