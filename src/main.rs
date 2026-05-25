use neuron_poc::guard::Guard;
use neuron_poc::memory::NeuronField;
use neuron_poc::queue::{EventPacket, EventQueue};

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

fn main() {
    println!("====================================================");
    println!("🧠 LLM-Guarded Event Engine PoC: Rhythm Tracker 🧠");
    println!("====================================================\n");

    // Initialize a NeuronField with 3 neurons:
    // Node 0: Input A (Correct Pattern, targets Node 2)
    // Node 1: Input B (Noise Pattern, targets Node 2)
    // Node 2: Detector Node (threshold = 1.0, targets 999)
    let field = NeuronField::new(3);

    unsafe {
        let n0 = field.get_neuron(0);
        n0.potential = 0.0;
        n0.threshold = 1.0;
        n0.target_id = 2;
        n0.weight = 1.5; // Start with weight >= 1.0

        let n1 = field.get_neuron(1);
        n1.potential = 0.0;
        n1.threshold = 1.0;
        n1.target_id = 2;
        n1.weight = 1.5; // Start with weight >= 1.0 (susceptible to noise)

        let n2 = field.get_neuron(2);
        n2.potential = 0.0;
        n2.threshold = 1.0;
        n2.target_id = 999; // End of chain
        n2.weight = 0.0;
    }

    println!("Initial State:");
    unsafe {
        println!("  Detector Threshold: 1.0");
        println!(
            "  W0 (Input A -> Detector): {:.4}",
            field.get_neuron(0).weight
        );
        println!(
            "  W1 (Input B -> Detector): {:.4} (Susceptible to Noise)",
            field.get_neuron(1).weight
        );
    }
    println!("\nGoal: Train the network so that:");
    println!("  - Input A (Correct Pattern) triggers the Detector.");
    println!("  - Input B (Noise Pattern) is filtered out (does NOT trigger Detector).");
    println!("  - This requires W1 to converge to < 1.0, while W0 remains >= 1.0.\n");

    println!("--- Starting Training Loop ---");

    let mut epoch = 0;
    loop {
        epoch += 1;
        // Alternately present Correct Pattern (Input A) and Noise Pattern (Input B)
        let is_correct_pattern = epoch % 2 == 1;

        if is_correct_pattern {
            // Correct Pattern: Trigger Input A (Node 0)
            // If it fires the Detector, we reinforce with positive feedback (+0.05)
            propagate_trainer(&field, 0, 1.0, None, 0.05);
        } else {
            // Noise Pattern: Trigger Input B (Node 1)
            // If it fires the Detector, this is a false positive! We penalize with negative feedback (-0.20)
            propagate_trainer(&field, 1, 1.0, None, -0.20);
        }

        unsafe {
            let w0 = field.get_neuron(0).weight;
            let w1 = field.get_neuron(1).weight;
            println!(
                "Epoch {:02}: Pattern = {}, W0 = {:.4}, W1 = {:.4}",
                epoch,
                if is_correct_pattern {
                    "Correct (Input A)"
                } else {
                    "Noise   (Input B)"
                },
                w0,
                w1
            );

            // Convergence check:
            // W0 must be >= 1.0 (to trigger Detector)
            // W1 must be < 1.0 (to NOT trigger Detector)
            if w0 >= 1.0 && w1 < 1.0 {
                println!("\n🎉 Convergence Win! 🎉");
                println!("The network successfully learned to filter out noise!");
                println!("Final Weights:");
                println!("  W0 (Input A -> Detector): {:.4}", w0);
                println!("  W1 (Input B -> Detector): {:.4}", w1);
                break;
            }
        }

        if epoch >= 50 {
            println!("\nTraining stopped after 50 epochs.");
            break;
        }
    }
    println!("====================================================");
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
