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
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RuntimeMode {
    Run,
    Trainer,
}

#[derive(Debug, Clone, Copy)]
pub struct EventPacket {
    pub target_id: u32,
    pub magnitude: f32,
    pub source_id: Option<u32>, // Only populated in Trainer Mode
}

pub struct EventQueue {
    pub sender: Sender<EventPacket>,
    pub receiver: Receiver<EventPacket>,
}

impl EventQueue {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        Self { sender, receiver }
    }

    pub fn push(&self, packet: EventPacket) {
        self.sender
            .send(packet)
            .expect("Failed to send event packet");
    }

    /// Spawns worker threads that run in the background.
    /// Returns a vector of JoinHandles so the caller can wait for them to finish.
    pub fn spawn_workers<F>(
        &self,
        neuron_field: Arc<NeuronField>,
        num_workers: usize,
        process_fn: F,
    ) -> Vec<JoinHandle<()>>
    where
        F: Fn(&NeuronField, EventPacket) + Send + Sync + 'static + Copy,
    {
        let mut handles = Vec::new();
        for _ in 0..num_workers {
            let receiver = self.receiver.clone();
            let field = Arc::clone(&neuron_field);
            let handle = thread::spawn(move || {
                while let Ok(packet) = receiver.recv() {
                    process_fn(&field, packet);
                }
            });
            handles.push(handle);
        }
        handles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::NeuronField;
    use rand::Rng;

    #[test]
    fn test_queue_push_and_receive() {
        let queue = EventQueue::new();
        queue.push(EventPacket {
            target_id: 5,
            magnitude: 1.5,
            source_id: None,
        });

        let packet = queue.receiver.recv().unwrap();
        assert_eq!(packet.target_id, 5);
        assert_eq!(packet.magnitude, 1.5);
        assert_eq!(packet.source_id, None);
    }

    #[test]
    fn test_thread_independence_stress() {
        // Phase 1 Checklist: Thread Independence
        // Spin up 4 worker threads. Blast 10,000 independent event packets at random neuron IDs
        // through the queue and verify that they are processed simultaneously without panics.
        let num_neurons = 1000;
        let neuron_field = Arc::new(NeuronField::new(num_neurons));
        let queue = EventQueue::new();

        // The process function simply updates the potential of the target neuron
        let process_fn = |field: &NeuronField, packet: EventPacket| {
            unsafe {
                let neuron = field.get_neuron(packet.target_id as usize);
                // Since multiple threads might write to the same neuron, we perform raw writes.
                // In a real network, we'd avoid collisions or handle them.
                // For the stress test, we just want to verify raw pointer writes from 4 threads
                // do not panic or crash.
                neuron.potential += packet.magnitude;
            }
        };

        let handles = queue.spawn_workers(Arc::clone(&neuron_field), 4, process_fn);

        // Blast 10,000 events
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let target_id = rng.gen_range(0..num_neurons) as u32;
            queue.push(EventPacket {
                target_id,
                magnitude: 0.1,
                source_id: None,
            });
        }

        // Drop the queue sender so the receiver loops terminate when the queue is empty
        drop(queue.sender);

        // Wait for all worker threads to finish
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify that some potentials were updated and no crashes occurred
        let mut total_potential = 0.0;
        for i in 0..num_neurons {
            unsafe {
                total_potential += neuron_field.get_neuron(i).potential;
            }
        }
        assert!(total_potential > 0.0);
        println!("Total potential after stress test: {}", total_potential);
    }
}
