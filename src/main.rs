pub mod guard;
pub mod memory;
pub mod queue;

use guard::Guard;
use memory::NeuronField;
use queue::{EventPacket, EventQueue};

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

        // Apply potential
        neuron.potential += magnitude;

        // Evaluate threshold
        if neuron.potential >= neuron.threshold {
            let target_id = neuron.target_id;
            let weight = neuron.weight;

            if target_id != 999 && (target_id as usize) < field.size {
                // Extend the Guard chain recursively
                propagate_trainer(
                    field,
                    target_id,
                    weight,
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

fn main() {
    println!("Hello, world! Welcome to Rust!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_mode_flight() {
        // Phase 2 Checklist: Run Mode Flight Test
        // Feed a spike cascade through a sequence of 5 nodes.
        // Verify that the event packet payload contains zero origin trackers,
        // and that execution flies forward sequentially.
        let field = NeuronField::new(5);
        let queue = EventQueue::new();

        // Initialize 5 nodes: Node 0 -> Node 1 -> Node 2 -> Node 3 -> Node 4
        unsafe {
            for i in 0..5 {
                let n = field.get_neuron(i);
                n.potential = 0.0;
                n.threshold = 1.0;
                n.target_id = (i + 1) as u32;
                n.weight = 1.0;
            }
            // Node 4 is the end of the chain
            field.get_neuron(4).target_id = 999;
        }

        // Push the initial event to Node 0
        queue.push(EventPacket {
            target_id: 0,
            magnitude: 1.0,
            source_id: None,
        });

        // Process exactly 5 events in the cascade sequentially.
        // This is extremely fast, deterministic, and tests the exact propagation path.
        for _ in 0..5 {
            let packet = queue.receiver.recv().unwrap();
            assert_eq!(packet.source_id, None); // Verify zero origin trackers
            propagate_run(&field, &queue, packet);
        }

        // Verify that the cascade reached Node 4 and reset potentials along the way
        unsafe {
            for i in 0..5 {
                assert_eq!(field.get_neuron(i).potential, 0.0);
            }
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
}
