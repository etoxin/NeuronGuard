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

pub mod guard;
pub mod memory;
pub mod neuron_guard;
pub mod queue;
pub mod run;
pub mod train;

#[cfg(feature = "extension-module")]
use crate::neuron_guard::{ParallelRouter, ThreadBoundedNeuronField};
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use std::sync::atomic::{AtomicI32, Ordering};
#[cfg(feature = "extension-module")]
use std::sync::Arc;

/// NeuronGuardField
/// The main Neuromorphic Cortex class exposed directly to Python runtimes.
#[cfg(feature = "extension-module")]
#[pyclass]
pub struct NeuronGuardField {
    sensory_count: usize,
    motor_count: usize,
    sensory_neurons: ThreadBoundedNeuronField,
    // The underlying atomic potentials modified concurrently across threads
    motor_potentials: Arc<Vec<AtomicI32>>,
    router: ParallelRouter,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl NeuronGuardField {
    /// Constructor exposed to Python: ng.NeuronGuardField(sensory_count, motor_count)
    #[new]
    fn new(sensory_count: usize, motor_count: usize) -> Self {
        let mut potentials = Vec::with_capacity(motor_count);
        for _ in 0..motor_count {
            potentials.push(AtomicI32::new(0));
        }

        let sensory_neurons = ThreadBoundedNeuronField::new(sensory_count);
        unsafe {
            for i in 0..sensory_count {
                let n = sensory_neurons.get_neuron(i);
                n.token_id = i as u32;
            }
        }

        let motor_potentials = Arc::new(potentials);
        let router = ParallelRouter::new(Arc::clone(&motor_potentials));

        NeuronGuardField {
            sensory_count,
            motor_count,
            sensory_neurons,
            motor_potentials,
            router,
        }
    }

    /// Process Stream
    /// Accepts a list of incoming raw sensory stimuli token IDs.
    /// Explicitly drops the GIL to unblock parallel crossbeam thread execution.
    fn process_stream(
        &self,
        py: Python,
        sensory_tokens: Vec<u32>,
        training_mode: bool,
    ) -> PyResult<u32> {
        // Drop the Python Global Interpreter Lock (GIL)
        py.allow_threads(|| {
            // ---- BARE-METAL CORRECTIONS / INFERENCE GO HERE ----
            // 1. Map tokens straight to cache offsets (token_id << 6)
            // 2. Dispatch work across the parallel thread pool using Thread IDs
            for &token_id in &sensory_tokens {
                if (token_id as usize) < self.sensory_count {
                    unsafe {
                        let neuron = self.sensory_neurons.get_neuron(token_id as usize);
                        self.router.broadcast(*neuron);
                    }
                }
            }

            // Give the background threads a moment to process the broadcasted neurons
            std::thread::sleep(std::time::Duration::from_millis(1));

            // Locate highest activated motor neuron index
            let mut highest_index = 0;
            let mut max_potential = i32::MIN;

            for i in 0..self.motor_count {
                let pot = self.motor_potentials[i].load(Ordering::Relaxed);
                if pot > max_potential {
                    max_potential = pot;
                    highest_index = i as u32;
                }
            }

            // 3. Apply Guard/Lease stack transformations if training_mode is true
            if training_mode {
                for &token_id in &sensory_tokens {
                    if (token_id as usize) < self.sensory_count {
                        if let Some(lease) =
                            self.sensory_neurons.try_acquire_lease(token_id as usize)
                        {
                            let neuron = lease.neuron();
                            neuron.update_or_add_connection(highest_index, 15);
                            for j in 0..neuron.active_connections as usize {
                                let target = neuron.target_neuron_ids[j];
                                if target != highest_index {
                                    neuron.weight_modifiers[j] =
                                        neuron.weight_modifiers[j].saturating_sub(5);
                                }
                            }
                        }
                    }
                }
            }

            Ok(highest_index)
        })
    }

    /// Tick Decay
    /// Exposes your prototype's background metabolic forgetting clock straight to the Python loop.
    fn tick_decay(&self, py: Python, decay_factor: f32) -> PyResult<()> {
        py.allow_threads(|| {
            for potential in self.motor_potentials.iter() {
                let current = potential.load(Ordering::Relaxed);
                if current != 0 {
                    let decayed = (current as f32 * decay_factor) as i32;
                    potential.store(decayed, Ordering::Relaxed);
                }
            }
            Ok(())
        })
    }
}

/// The root Python Module Definition
#[cfg(feature = "extension-module")]
#[pymodule]
fn neuronguard(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuronGuardField>()?;
    Ok(())
}
