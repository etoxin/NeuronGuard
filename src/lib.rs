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
use crate::neuron_guard::ThreadBoundedNeuronField;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use rayon::prelude::*;

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

        NeuronGuardField {
            sensory_count,
            motor_count,
            sensory_neurons,
            motor_potentials,
        }
    }

    /// Predict
    /// Evaluates the active sensory tokens synchronously on the calling thread,
    /// adding their weights directly to the motor potentials.
    /// This is extremely fast and perfect for batch evaluation.
    fn predict(&self, py: Python, sensory_tokens: Vec<u32>) -> PyResult<u32> {
        py.allow_threads(|| {
            for &token_id in &sensory_tokens {
                if (token_id as usize) < self.sensory_count {
                    unsafe {
                        let n = self.sensory_neurons.get_neuron(token_id as usize);
                        for i in 0..n.active_connections as usize {
                            let target = n.target_neuron_ids[i] as usize;
                            if target < self.motor_count {
                                self.motor_potentials[target]
                                    .fetch_add(n.weight_modifiers[i] as i32, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }

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

            Ok(highest_index)
        })
    }

    /// Predict Batch
    /// Evaluates a batch of sensory token streams in parallel using rayon.
    fn predict_batch(&self, py: Python, batch_tokens: Vec<Vec<u32>>) -> PyResult<Vec<u32>> {
        py.allow_threads(|| {
            let results: Vec<u32> = batch_tokens
                .par_iter()
                .map(|sensory_tokens| {
                    let mut local_potentials = vec![0i32; self.motor_count];

                    for &token_id in sensory_tokens {
                        if (token_id as usize) < self.sensory_count {
                            unsafe {
                                let n = self.sensory_neurons.get_neuron(token_id as usize);
                                for i in 0..n.active_connections as usize {
                                    let target = n.target_neuron_ids[i] as usize;
                                    if target < self.motor_count {
                                        local_potentials[target] += n.weight_modifiers[i] as i32;
                                    }
                                }
                            }
                        }
                    }

                    let mut highest_index = 0;
                    let mut max_potential = i32::MIN;

                    for (i, &pot) in local_potentials.iter().enumerate() {
                        if pot > max_potential {
                            max_potential = pot;
                            highest_index = i as u32;
                        }
                    }
                    highest_index
                })
                .collect();
            Ok(results)
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

    /// Train Stream
    /// Trains active sensory tokens to target a specific correct motor neuron ID.
    fn train_stream(
        &self,
        py: Python,
        sensory_tokens: Vec<u32>,
        correct_motor_id: u32,
        amplify_delta: i16,
        suppress_delta: i16,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            for &token_id in &sensory_tokens {
                if (token_id as usize) < self.sensory_count {
                    if let Some(lease) = self.sensory_neurons.try_acquire_lease(token_id as usize) {
                        let neuron = lease.neuron();
                        // Amplify correct expert pathway
                        neuron.update_or_add_connection(correct_motor_id, amplify_delta);

                        // Suppress incorrect expert pathways
                        for j in 0..neuron.active_connections as usize {
                            let target = neuron.target_neuron_ids[j];
                            if target != correct_motor_id {
                                neuron.weight_modifiers[j] =
                                    neuron.weight_modifiers[j].saturating_sub(suppress_delta);
                            }
                        }
                    }
                }
            }
            Ok(())
        })
    }

    /// Train Batch
    /// Trains a batch of (sensory_tokens, correct_motor_id) in parallel using rayon.
    fn train_batch(
        &self,
        py: Python,
        batch: Vec<(Vec<u32>, u32)>,
        amplify_delta: i16,
        suppress_delta: i16,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            batch.par_iter().for_each(|(sensory_tokens, correct_motor_id)| {
                for &token_id in sensory_tokens {
                    if (token_id as usize) < self.sensory_count {
                        if let Some(lease) = self.sensory_neurons.try_acquire_lease(token_id as usize) {
                            let neuron = lease.neuron();
                            neuron.update_or_add_connection(*correct_motor_id, amplify_delta);

                            for j in 0..neuron.active_connections as usize {
                                let target = neuron.target_neuron_ids[j];
                                if target != *correct_motor_id {
                                    neuron.weight_modifiers[j] =
                                        neuron.weight_modifiers[j].saturating_sub(suppress_delta);
                                }
                            }
                        }
                    }
                }
            });
            Ok(())
        })
    }

    /// Reset Potentials
    /// Resets all motor neuron potentials to zero.
    fn reset_potentials(&self, py: Python) -> PyResult<()> {
        py.allow_threads(|| {
            for potential in self.motor_potentials.iter() {
                potential.store(0, Ordering::Relaxed);
            }
            Ok(())
        })
    }

    /// Get Potentials
    /// Returns the current potentials of all motor neurons.
    fn get_potentials(&self, py: Python) -> PyResult<Vec<i32>> {
        py.allow_threads(|| {
            let mut potentials = Vec::with_capacity(self.motor_count);
            for potential in self.motor_potentials.iter() {
                potentials.push(potential.load(Ordering::Relaxed));
            }
            Ok(potentials)
        })
    }

    /// Get Neuron Synapses
    /// Introspection method to read the explicit synaptic weights of a sensory neuron.
    /// Returns a list of (target_motor_id, weight) tuples.
    fn get_neuron_synapses(&self, token_id: u32) -> PyResult<Vec<(u32, i16)>> {
        if (token_id as usize) >= self.sensory_count {
            return Ok(vec![]);
        }
        let mut synapses = Vec::new();
        unsafe {
            let n = self.sensory_neurons.get_neuron(token_id as usize);
            for i in 0..n.active_connections as usize {
                synapses.push((n.target_neuron_ids[i], n.weight_modifiers[i]));
            }
        }
        Ok(synapses)
    }

    /// Save Weights
    /// Serializes and saves the sensory neurons' raw memory to a binary file.
    fn save_weights(&self, py: Python, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.sensory_neurons.storage as *const u8,
                    self.sensory_count * 64,
                )
            };
            std::fs::write(path, bytes)?;
            Ok(())
        })
    }

    /// Load Weights
    /// Loads and memory-maps the sensory neurons' connections from a binary file for zero-copy access.
    fn load_weights(&mut self, py: Python, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let file = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
            let mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };
            self.sensory_neurons = ThreadBoundedNeuronField::from_mmap(mmap, self.sensory_count);
            Ok(())
        })
    }
}

#[cfg(feature = "extension-module")]
fn stem(word: &str) -> String {
    if word.len() <= 4 {
        return word.to_string();
    }
    if word.ends_with("tion") {
        return word[..word.len() - 4].to_string();
    }
    if word.ends_with("sion") {
        return word[..word.len() - 4].to_string();
    }
    if word.ends_with("ment") {
        return word[..word.len() - 4].to_string();
    }
    if word.ends_with("ness") {
        return word[..word.len() - 4].to_string();
    }
    if word.ends_with("ing") && word.len() > 5 {
        return word[..word.len() - 3].to_string();
    }
    if word.ends_with("ies") && word.len() > 4 {
        return format!("{}y", &word[..word.len() - 3]);
    }
    if word.ends_with("ly") && word.len() > 4 {
        return word[..word.len() - 2].to_string();
    }
    if word.ends_with("ed") && word.len() > 4 {
        return word[..word.len() - 2].to_string();
    }
    if word.ends_with("es") && word.len() > 4 {
        return word[..word.len() - 2].to_string();
    }
    if word.ends_with("s") && !word.ends_with("ss") && word.len() > 4 {
        return word[..word.len() - 1].to_string();
    }
    word.to_string()
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn tokenize(text: String, stop_words: std::collections::HashSet<String>, apply_stemming: bool, min_length: usize) -> Vec<String> {
    let mut tokens = Vec::new();
    let lower = text.to_lowercase();
    for token in lower.split(|c: char| !c.is_alphanumeric()) {
        if token.len() >= min_length && !stop_words.contains(token) {
            if apply_stemming {
                tokens.push(stem(token));
            } else {
                tokens.push(token.to_string());
            }
        }
    }
    tokens
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn neuronguard(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuronGuardField>()?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    Ok(())
}
