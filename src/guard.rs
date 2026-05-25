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

pub struct Guard<'a> {
    pub neuron_id: u32,
    pub field: &'a NeuronField,
    pub parent: Option<&'a Guard<'a>>,
}

impl<'a> Guard<'a> {
    /// Creates a new transactional Guard scope for a neuron.
    pub fn new(neuron_id: u32, field: &'a NeuronField, parent: Option<&'a Guard<'a>>) -> Self {
        Self {
            neuron_id,
            field,
            parent,
        }
    }

    /// Traverses the open session trace backwards and adjusts the weights of the neurons
    /// that contributed to this cascade.
    pub fn propagate_feedback(&self, feedback: f32) {
        let mut current = self.parent;
        while let Some(guard) = current {
            unsafe {
                let neuron = self.field.get_neuron(guard.neuron_id as usize);
                // Adjust the weight of the contributing neuron
                neuron.weight += feedback;
            }
            current = guard.parent;
        }
    }
}

// Implement Drop to automatically reset the neuron's potential when the guard goes out of scope.
// This "primes" the memory block for the next transaction/thread.
impl<'a> Drop for Guard<'a> {
    fn drop(&mut self) {
        unsafe {
            let neuron = self.field.get_neuron(self.neuron_id as usize);
            neuron.potential = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_feedback_and_automatic_drop() {
        let field = NeuronField::new(3);

        // Initialize 3 neurons: Node 0 -> Node 1 -> Node 2
        unsafe {
            let n0 = field.get_neuron(0);
            n0.potential = 1.0;
            n0.threshold = 1.5;
            n0.target_id = 1;
            n0.weight = 0.5;

            let n1 = field.get_neuron(1);
            n1.potential = 1.0;
            n1.threshold = 1.5;
            n1.target_id = 2;
            n1.weight = 0.5;

            let n2 = field.get_neuron(2);
            n2.potential = 1.0;
            n2.threshold = 1.5;
            n2.target_id = 999; // End of chain
            n2.weight = 0.0;
        }

        // Create the transactional guard scope
        {
            let guard0 = Guard::new(0, &field, None);
            let guard1 = Guard::new(1, &field, Some(&guard0));
            let guard2 = Guard::new(2, &field, Some(&guard1));

            // Propagate feedback backwards from Node 2
            guard2.propagate_feedback(0.25);

            // Verify that Node 0 and Node 1 weights were updated
            unsafe {
                assert_eq!(field.get_neuron(0).weight, 0.75); // 0.5 + 0.25
                assert_eq!(field.get_neuron(1).weight, 0.75); // 0.5 + 0.25
                assert_eq!(field.get_neuron(2).weight, 0.0); // Unchanged
            }

            // Verify potentials are still active inside the scope
            unsafe {
                assert_eq!(field.get_neuron(0).potential, 1.0);
                assert_eq!(field.get_neuron(1).potential, 1.0);
                assert_eq!(field.get_neuron(2).potential, 1.0);
            }
        } // <-- All guards go out of scope and drop here!

        // Verify that potentials were automatically reset to 0.0 upon drop
        unsafe {
            assert_eq!(field.get_neuron(0).potential, 0.0);
            assert_eq!(field.get_neuron(1).potential, 0.0);
            assert_eq!(field.get_neuron(2).potential, 0.0);
        }
    }
}
