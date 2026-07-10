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

pub mod neuron_guard;

#[cfg(feature = "extension-module")]
use crate::neuron_guard::ThreadBoundedNeuronField;
#[cfg(feature = "extension-module")]
use crate::neuron_guard::MAX_THREADS;
#[cfg(feature = "extension-module")]
use parking_lot::Mutex;
#[cfg(feature = "extension-module")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use rayon::prelude::*;

#[cfg(feature = "extension-module")]
use std::collections::BTreeMap;

/// NeuronGuardField
/// The main Neuromorphic Cortex class exposed directly to Python runtimes.
#[cfg(feature = "extension-module")]
#[pyclass]
pub struct NeuronGuardField {
    sensory_count: usize,
    motor_count: usize,
    sensory_neurons: ThreadBoundedNeuronField,
    // Backward-compatible snapshot for reset/get_potentials. Predictions compute
    // request-local scores, so concurrent calls cannot contaminate one another.
    last_potentials: Mutex<Vec<i32>>,
}

#[cfg(feature = "extension-module")]
impl NeuronGuardField {
    fn score_tokens(&self, sensory_tokens: &[u32]) -> Vec<i32> {
        let mut potentials = vec![0i32; self.motor_count];
        for &token_id in sensory_tokens {
            self.sensory_neurons
                .with_neuron(token_id as usize, |neuron| {
                    for connection in 0..neuron.active_connections as usize {
                        let target = neuron.target_neuron_ids[connection] as usize;
                        if target < self.motor_count {
                            potentials[target] += neuron.weight_modifiers[connection] as i32;
                        }
                    }
                });
        }
        potentials
    }

    fn highest_score(potentials: &[i32]) -> u32 {
        potentials
            .iter()
            .enumerate()
            .fold((0usize, i32::MIN), |best, (index, &potential)| {
                if potential > best.1 {
                    (index, potential)
                } else {
                    best
                }
            })
            .0 as u32
    }

    fn update_neuron(
        neuron: &mut crate::neuron_guard::ThreadBoundedNeuron,
        correct_motor_id: u32,
        amplify_delta: i16,
        suppress_delta: i16,
    ) {
        neuron.update_or_add_connection(correct_motor_id, amplify_delta);
        for connection in 0..neuron.active_connections as usize {
            if neuron.target_neuron_ids[connection] != correct_motor_id {
                neuron.weight_modifiers[connection] =
                    neuron.weight_modifiers[connection].saturating_sub(suppress_delta);
            }
        }
    }

    fn validate_tokens(&self, sensory_tokens: &[u32]) -> PyResult<()> {
        if let Some(token) = sensory_tokens
            .iter()
            .find(|&&token| token as usize >= self.sensory_count)
        {
            return Err(PyValueError::new_err(format!(
                "sensory token {token} is outside 0..{}",
                self.sensory_count
            )));
        }
        Ok(())
    }

    fn validate_motor(&self, motor_id: u32) -> PyResult<()> {
        if motor_id as usize >= self.motor_count {
            return Err(PyValueError::new_err(format!(
                "motor id {motor_id} is outside 0..{}",
                self.motor_count
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl NeuronGuardField {
    /// Constructor exposed to Python: ng.NeuronGuardField(sensory_count, motor_count)
    #[new]
    fn new(sensory_count: usize, motor_count: usize) -> PyResult<Self> {
        if sensory_count == 0 {
            return Err(PyValueError::new_err("sensory_count must be positive"));
        }
        if motor_count == 0 || motor_count > MAX_THREADS {
            return Err(PyValueError::new_err(format!(
                "motor_count must be between 1 and {MAX_THREADS}"
            )));
        }

        Ok(NeuronGuardField {
            sensory_count,
            motor_count,
            sensory_neurons: ThreadBoundedNeuronField::new(sensory_count),
            last_potentials: Mutex::new(vec![0; motor_count]),
        })
    }

    /// Predict
    /// Evaluates the active sensory tokens synchronously on the calling thread,
    /// adding their weights directly to the motor potentials.
    /// This is extremely fast and perfect for batch evaluation.
    fn predict(&self, py: Python, sensory_tokens: Vec<u32>) -> PyResult<u32> {
        self.validate_tokens(&sensory_tokens)?;
        py.allow_threads(|| {
            let potentials = self.score_tokens(&sensory_tokens);
            let prediction = Self::highest_score(&potentials);
            *self.last_potentials.lock() = potentials;
            Ok(prediction)
        })
    }

    /// Predict Scores
    /// Atomically computes and returns request-local scores for all motor neurons.
    fn predict_scores(&self, py: Python, sensory_tokens: Vec<u32>) -> PyResult<Vec<i32>> {
        self.validate_tokens(&sensory_tokens)?;
        py.allow_threads(|| {
            let potentials = self.score_tokens(&sensory_tokens);
            *self.last_potentials.lock() = potentials.clone();
            Ok(potentials)
        })
    }

    /// Predict Batch
    /// Evaluates a batch of sensory token streams in parallel using rayon.
    fn predict_batch(&self, py: Python, batch_tokens: Vec<Vec<u32>>) -> PyResult<Vec<u32>> {
        for tokens in &batch_tokens {
            self.validate_tokens(tokens)?;
        }
        py.allow_threads(|| {
            let results: Vec<u32> = batch_tokens
                .par_iter()
                .map(|tokens| Self::highest_score(&self.score_tokens(tokens)))
                .collect();
            Ok(results)
        })
    }

    /// Tick Decay
    /// Exposes your prototype's background metabolic forgetting clock straight to the Python loop.
    fn tick_decay(&self, py: Python, decay_factor: f32) -> PyResult<()> {
        py.allow_threads(|| {
            let mut potentials = self.last_potentials.lock();
            for potential in potentials.iter_mut() {
                *potential = (*potential as f32 * decay_factor) as i32;
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
        self.validate_motor(correct_motor_id)?;
        self.validate_tokens(&sensory_tokens)?;
        py.allow_threads(|| {
            for &token_id in &sensory_tokens {
                self.sensory_neurons
                    .with_neuron_mut(token_id as usize, |neuron| {
                        Self::update_neuron(neuron, correct_motor_id, amplify_delta, suppress_delta)
                    });
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
        for (tokens, motor_id) in &batch {
            self.validate_motor(*motor_id)?;
            self.validate_tokens(tokens)?;
        }
        py.allow_threads(|| {
            let mut updates_by_token: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
            for (tokens, motor_id) in &batch {
                for &token in tokens {
                    updates_by_token.entry(token).or_default().push(*motor_id);
                }
            }
            updates_by_token.par_iter().for_each(|(&token, motor_ids)| {
                self.sensory_neurons
                    .with_neuron_mut(token as usize, |neuron| {
                        for &motor_id in motor_ids {
                            Self::update_neuron(neuron, motor_id, amplify_delta, suppress_delta);
                        }
                    });
            });
            Ok(())
        })
    }

    /// Reset Potentials
    /// Resets all motor neuron potentials to zero.
    fn reset_potentials(&self, py: Python) -> PyResult<()> {
        py.allow_threads(|| {
            self.last_potentials.lock().fill(0);
            Ok(())
        })
    }

    /// Get Potentials
    /// Returns the current potentials of all motor neurons.
    fn get_potentials(&self, py: Python) -> PyResult<Vec<i32>> {
        py.allow_threads(|| Ok(self.last_potentials.lock().clone()))
    }

    /// Get Neuron Synapses
    /// Introspection method to read the explicit synaptic weights of a sensory neuron.
    /// Returns a list of (target_motor_id, weight) tuples.
    fn get_neuron_synapses(&self, token_id: u32) -> PyResult<Vec<(u32, i16)>> {
        if (token_id as usize) >= self.sensory_count {
            return Ok(vec![]);
        }
        Ok(self
            .sensory_neurons
            .with_neuron(token_id as usize, |neuron| {
                (0..neuron.active_connections as usize)
                    .map(|index| {
                        (
                            neuron.target_neuron_ids[index],
                            neuron.weight_modifiers[index],
                        )
                    })
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Save Weights
    /// Serializes and saves the sensory neurons' raw memory to a binary file.
    fn save_weights(&self, py: Python, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let bytes = self.sensory_neurons.snapshot_bytes();
            std::fs::write(path, bytes)?;
            Ok(())
        })
    }

    /// Load Weights
    /// Loads and memory-maps the sensory neurons' connections from a binary file for zero-copy access.
    fn load_weights(&mut self, py: Python, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)?;
            let expected_len = ThreadBoundedNeuronField::byte_len_for(self.sensory_count);
            let actual_len = file.metadata()?.len() as usize;
            if actual_len != expected_len {
                return Err(PyValueError::new_err(format!(
                    "weight file is {actual_len} bytes; expected {expected_len}"
                )));
            }
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
    for suffix in ["tion", "sion", "ment", "ness"] {
        if let Some(stripped) = word.strip_suffix(suffix) {
            return stripped.to_string();
        }
    }
    if let Some(stripped) = word.strip_suffix("ing").filter(|_| word.len() > 5) {
        return stripped.to_string();
    }
    if let Some(stripped) = word.strip_suffix("ies").filter(|_| word.len() > 4) {
        return format!("{stripped}y");
    }
    if let Some(stripped) = word.strip_suffix("ly").filter(|_| word.len() > 4) {
        return stripped.to_string();
    }
    if let Some(stripped) = word.strip_suffix("ed").filter(|_| word.len() > 4) {
        return stripped.to_string();
    }
    if let Some(stripped) = word.strip_suffix("es").filter(|_| word.len() > 4) {
        return stripped.to_string();
    }
    if !word.ends_with("ss") {
        if let Some(stripped) = word.strip_suffix('s').filter(|_| word.len() > 4) {
            return stripped.to_string();
        }
    }
    word.to_string()
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn tokenize(
    text: String,
    stop_words: std::collections::HashSet<String>,
    apply_stemming: bool,
    min_length: usize,
) -> Vec<String> {
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
