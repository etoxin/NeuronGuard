// Copyright 2026 Adam Lusted
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::memory::NeuronField;
use crate::neuron_guard::ThreadBoundedNeuronField;
use crate::queue::{EventPacket, EventQueue};

/// Fast-path signal propagation for Run Mode.
/// This function is designed to be lightning-fast, lock-free, and unidirectional.
pub fn propagate_run(field: &NeuronField, queue: &EventQueue, packet: EventPacket) {
    unsafe {
        let neuron = field.get_neuron(packet.target_id as usize);
        neuron.potential += packet.magnitude;

        if neuron.potential >= neuron.threshold {
            neuron.potential = 0.0; // Reset potential on fire

            // Only propagate if there is a valid target
            if neuron.target_id != 999 && (neuron.target_id as usize) < field.size {
                let next_packet = EventPacket {
                    target_id: neuron.target_id,
                    magnitude: neuron.weight,
                    source_id: None, // Zero origin tracking in Run Mode
                };
                queue.push(next_packet);
            }
        }
    }
}

/// Evaluates a ThreadBoundedNeuron's active connections and accumulates their weights.
pub fn evaluate_neuron_potentials(
    field: &ThreadBoundedNeuronField,
    neuron_id: usize,
    vocab_size: usize,
    field_size: usize,
    expert_potentials: &mut [i32],
) {
    unsafe {
        let n = field.get_neuron(neuron_id);
        for i in 0..n.active_connections as usize {
            let target = n.target_neuron_ids[i] as usize;
            if target >= vocab_size && target < field_size {
                let expert_idx = target - vocab_size;
                if expert_idx < expert_potentials.len() {
                    expert_potentials[expert_idx] += n.weight_modifiers[i] as i32;
                }
            }
        }
    }
}

/// Simple text tokenizer and cleaner.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_evaluate_neuron_potentials() {
        let vocab_size = 10;
        let field_size = 12;
        let field = ThreadBoundedNeuronField::new(field_size);

        unsafe {
            let n = field.get_neuron(2);
            n.active_connections = 2;
            n.target_neuron_ids[0] = 10;
            n.weight_modifiers[0] = 15;
            n.target_neuron_ids[1] = 11;
            n.weight_modifiers[1] = -5;
        }

        let mut expert_potentials = [0i32; 2];
        evaluate_neuron_potentials(&field, 2, vocab_size, field_size, &mut expert_potentials);

        assert_eq!(expert_potentials[0], 15);
        assert_eq!(expert_potentials[1], -5);
    }
}
