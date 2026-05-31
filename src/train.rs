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

use crate::guard::Guard;
use crate::memory::NeuronField;
use crate::neuron_guard::ThreadBoundedNeuronField;

/// Transactional signal propagation for Trainer Mode.
/// This function runs synchronously on a single thread's stack, building a chain of Guards.
pub fn propagate_trainer(
    field: &NeuronField,
    neuron_id: u32,
    magnitude: f32,
    parent_guard: Option<&Guard>,
    feedback_value: f32,
) {
    unsafe {
        let neuron = field.get_neuron(neuron_id as usize);

        // Allocate temporary Guard scope on the stack
        let current_guard = Guard::new(neuron_id, field, parent_guard);

        // Apply potential multiplied by the incoming weight (if there is a parent)
        // For the root node, we apply the raw input magnitude.
        let incoming_signal = if parent_guard.is_some() {
            let parent_neuron = field.get_neuron(parent_guard.unwrap().neuron_id as usize);
            magnitude * parent_neuron.weight
        } else {
            magnitude
        };

        neuron.potential += incoming_signal;

        // Evaluate threshold
        if neuron.potential >= neuron.threshold {
            let target_id = neuron.target_id;

            if target_id != 999 && (target_id as usize) < field.size {
                // Extend the Guard chain recursively
                propagate_trainer(
                    field,
                    target_id,
                    magnitude, // Pass the original magnitude forward
                    Some(&current_guard),
                    feedback_value,
                );
            } else {
                // Reached the end of the cascade! Evaluate outcome and propagate feedback backwards.
                current_guard.propagate_feedback(feedback_value);
            }
        }
        // current_guard automatically drops here, resetting potential to 0.0
    }
}

/// Trains a connection on a ThreadBoundedNeuron using the Guard/Lease pattern.
/// Amplifies the correct expert pathway and suppresses incorrect expert pathways.
pub fn train_neuron_connection(
    field: &ThreadBoundedNeuronField,
    neuron_id: usize,
    correct_expert: u32,
    vocab_size: usize,
    amplify_delta: i16,
    suppress_delta: i16,
) {
    if let Some(lease) = field.try_acquire_lease(neuron_id) {
        let neuron = lease.neuron();

        // Amplify correct expert pathway
        neuron.update_or_add_connection(correct_expert, amplify_delta);

        // Suppress incorrect expert pathways
        for j in 0..neuron.active_connections as usize {
            let target = neuron.target_neuron_ids[j];
            if target != correct_expert && target >= vocab_size as u32 {
                neuron.weight_modifiers[j] =
                    neuron.weight_modifiers[j].saturating_sub(suppress_delta);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_neuron_connection() {
        let vocab_size = 10;
        let field_size = 12;
        let field = ThreadBoundedNeuronField::new(field_size);

        unsafe {
            let n = field.get_neuron(2);
            n.active_connections = 2;
            n.target_neuron_ids[0] = 10;
            n.weight_modifiers[0] = 15;
            n.target_neuron_ids[1] = 11;
            n.weight_modifiers[1] = 20;
        }

        // Train neuron 2 to target expert 10 (index 10)
        train_neuron_connection(&field, 2, 10, vocab_size, 5, 15);

        unsafe {
            let n = field.get_neuron(2);
            // Expert 10 should be amplified: 15 + 5 = 20
            assert_eq!(n.weight_modifiers[0], 20);
            // Expert 11 should be suppressed: 20 - 15 = 5
            assert_eq!(n.weight_modifiers[1], 5);
        }
    }

    #[test]
    fn test_trainer_mode_guard() {
        // Phase 2 Checklist: Trainer Mode Guard Test
        // Trigger a cascade where Node 0 activates Node 1, which activates Node 2.
        // Verify that Node 2 successfully passes a feedback signal backwards through
        // the open session trace to update Node 0's weight variable before the temporary thread ends.
        let field = NeuronField::new(3);

        // Initialize 3 nodes: Node 0 -> Node 1 -> Node 2
        unsafe {
            let n0 = field.get_neuron(0);
            n0.potential = 0.0;
            n0.threshold = 1.0;
            n0.target_id = 1;
            n0.weight = 1.0;

            let n1 = field.get_neuron(1);
            n1.potential = 0.0;
            n1.threshold = 1.0;
            n1.target_id = 2;
            n1.weight = 1.0;

            let n2 = field.get_neuron(2);
            n2.potential = 0.0;
            n2.threshold = 1.0;
            n2.target_id = 999; // End of chain
            n2.weight = 1.0;
        }

        // Trigger the cascade in Trainer Mode starting at Node 0 with feedback +0.5
        propagate_trainer(&field, 0, 1.0, None, 0.5);

        // Verify that Node 0's and Node 1's weights were updated by the feedback
        unsafe {
            assert_eq!(field.get_neuron(0).weight, 1.5); // 1.0 + 0.5
            assert_eq!(field.get_neuron(1).weight, 1.5); // 1.0 + 0.5
            assert_eq!(field.get_neuron(2).weight, 1.0); // Node 2 is the end, weight unchanged
        }

        // Verify that all potentials were automatically reset to 0.0 upon Guard drop
        unsafe {
            assert_eq!(field.get_neuron(0).potential, 0.0);
            assert_eq!(field.get_neuron(1).potential, 0.0);
            assert_eq!(field.get_neuron(2).potential, 0.0);
        }
    }

    #[test]
    fn test_rhythm_tracker_convergence() {
        // Phase 3 Checklist: The Convergence Win
        // Verify that the target node's weight changes until it consistently filters out noise.
        let field = NeuronField::new(3);

        unsafe {
            let n0 = field.get_neuron(0);
            n0.potential = 0.0;
            n0.threshold = 1.0;
            n0.target_id = 2;
            n0.weight = 1.5;

            let n1 = field.get_neuron(1);
            n1.potential = 0.0;
            n1.threshold = 1.0;
            n1.target_id = 2;
            n1.weight = 1.5;

            let n2 = field.get_neuron(2);
            n2.potential = 0.0;
            n2.threshold = 1.0;
            n2.target_id = 999;
            n2.weight = 0.0;
        }

        let mut converged = false;
        for epoch in 1..=50 {
            let is_correct_pattern = epoch % 2 == 1;
            if is_correct_pattern {
                propagate_trainer(&field, 0, 1.0, None, 0.05);
            } else {
                propagate_trainer(&field, 1, 1.0, None, -0.20);
            }

            unsafe {
                let w0 = field.get_neuron(0).weight;
                let w1 = field.get_neuron(1).weight;
                if w0 >= 1.0 && w1 < 1.0 {
                    converged = true;
                    break;
                }
            }
        }

        assert!(converged, "Failed to converge within 50 epochs");
    }
}
