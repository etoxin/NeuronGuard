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

pub mod attention;
pub mod cpg;
pub mod ensemble_mesh;
pub mod gen_memory;
pub mod guard;
pub mod memory;
pub mod neuron_guard;
pub mod queue;
pub mod run;
pub mod train;
pub mod train_gen;
pub mod wta;

#[cfg(feature = "extension-module")]
use crate::attention::SpikingAttentionState;
#[cfg(feature = "extension-module")]
use crate::ensemble_mesh::{InsectoidSimulation, PermanentSpatiotemporalEnsembleMesh};
#[cfg(feature = "extension-module")]
use crate::gen_memory::HighDensityNeuromorphicLine;
#[cfg(feature = "extension-module")]
use crate::neuron_guard::{ParallelRouter, ThreadBoundedNeuronField};
#[cfg(feature = "extension-module")]
use crate::train_gen::{base64_encode, NeuronGuardTrainerField};
#[cfg(feature = "extension-module")]
use crate::wta::HierarchicalWinnerTakeAll;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use std::collections::HashMap;
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

    /// Process Stream Synchronous
    /// Evaluates the active sensory tokens synchronously on the calling thread,
    /// adding their weights directly to the motor potentials.
    /// This is extremely fast and perfect for batch evaluation (no thread pool or sleep overhead).
    fn process_stream_sync(&self, py: Python, sensory_tokens: Vec<u32>) -> PyResult<u32> {
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

    /// Save Weights
    /// Serializes and saves the sensory neurons' connections to a binary file.
    fn save_weights(&self, py: Python, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let mut bytes = Vec::new();
            for i in 0..self.sensory_count {
                unsafe {
                    let n = self.sensory_neurons.get_neuron(i);
                    bytes.extend_from_slice(&n.active_connections.to_le_bytes());
                    for &target in &n.target_neuron_ids {
                        bytes.extend_from_slice(&target.to_le_bytes());
                    }
                    for &weight in &n.weight_modifiers {
                        bytes.extend_from_slice(&weight.to_le_bytes());
                    }
                }
            }
            std::fs::write(path, bytes)?;
            Ok(())
        })
    }

    /// Load Weights
    /// Loads and deserializes the sensory neurons' connections from a binary file.
    fn load_weights(&self, py: Python, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let bytes = std::fs::read(path)?;
            let mut offset = 0;
            for i in 0..self.sensory_count {
                if offset + 4 + 32 + 16 > bytes.len() {
                    break;
                }
                unsafe {
                    let n = self.sensory_neurons.get_neuron(i);
                    n.active_connections =
                        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                    offset += 4;
                    for j in 0..8 {
                        n.target_neuron_ids[j] =
                            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        offset += 4;
                    }
                    for j in 0..8 {
                        n.weight_modifiers[j] =
                            i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                        offset += 2;
                    }
                }
            }
            Ok(())
        })
    }
}

/// The root Python Module Definition
/// The root Python Module Definition
#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyPermanentSpatiotemporalEnsembleMesh {
    pub mesh: Arc<PermanentSpatiotemporalEnsembleMesh>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyPermanentSpatiotemporalEnsembleMesh {
    #[new]
    fn new(size_per_field: usize) -> Self {
        Self {
            mesh: Arc::new(PermanentSpatiotemporalEnsembleMesh::new(size_per_field)),
        }
    }

    #[pyo3(signature = (tokens, training_mode, correct_target=None, decay_factor=0.9))]
    fn process_frame(
        &self,
        py: Python,
        tokens: [u32; 3],
        training_mode: bool,
        correct_target: Option<u32>,
        decay_factor: f32,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.mesh
                .process_frame(tokens, training_mode, correct_target, decay_factor);
            Ok(())
        })
    }

    fn decay(&self, py: Python, decay_factor: f32) -> PyResult<()> {
        py.allow_threads(|| {
            self.mesh.decay(decay_factor);
            Ok(())
        })
    }

    fn get_active_nodes(&self, py: Python) -> PyResult<Vec<u32>> {
        py.allow_threads(|| Ok(self.mesh.get_active_nodes()))
    }

    fn get_loop_intensities(&self, py: Python) -> PyResult<HashMap<u32, i32>> {
        py.allow_threads(|| Ok(self.mesh.get_loop_intensities()))
    }

    fn get_structural_mutations(&self, py: Python) -> PyResult<Vec<(u32, u32, u32, u64)>> {
        py.allow_threads(|| {
            let mutations = self.mesh.get_structural_mutations();
            let py_mutations = mutations
                .into_iter()
                .map(|m| (m.token_id, m.evicted_target, m.new_target, m.timestamp_us))
                .collect();
            Ok(py_mutations)
        })
    }
}

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyInsectoidSimulation {
    pub sim: InsectoidSimulation,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyInsectoidSimulation {
    #[new]
    fn new() -> Self {
        Self {
            sim: InsectoidSimulation::new(),
        }
    }

    #[pyo3(signature = (training_mode, correct_target=None, decay_factor=0.9))]
    fn step(
        &mut self,
        py: Python,
        training_mode: bool,
        correct_target: Option<u32>,
        decay_factor: f32,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.sim.step(training_mode, correct_target, decay_factor);
            Ok(())
        })
    }

    fn get_phases(&self) -> Vec<f32> {
        self.sim.phases.to_vec()
    }

    fn get_velocities(&self) -> Vec<f32> {
        self.sim.velocities.to_vec()
    }

    fn get_active_nodes(&self, py: Python) -> PyResult<Vec<u32>> {
        py.allow_threads(|| Ok(self.sim.mesh.get_active_nodes()))
    }

    fn get_loop_intensities(&self, py: Python) -> PyResult<HashMap<u32, i32>> {
        py.allow_threads(|| Ok(self.sim.mesh.get_loop_intensities()))
    }

    fn get_structural_mutations(&self, py: Python) -> PyResult<Vec<(u32, u32, u32, u64)>> {
        py.allow_threads(|| {
            let mutations = self.sim.mesh.get_structural_mutations();
            let py_mutations = mutations
                .into_iter()
                .map(|m| (m.token_id, m.evicted_target, m.new_target, m.timestamp_us))
                .collect();
            Ok(py_mutations)
        })
    }
}

#[cfg(feature = "extension-module")]
#[pyclass]
#[derive(Clone)]
pub struct PyPermanentNeuromorphicLine {
    pub line: HighDensityNeuromorphicLine,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyPermanentNeuromorphicLine {
    #[new]
    fn new() -> Self {
        Self {
            line: HighDensityNeuromorphicLine::new(15),
        }
    }

    #[getter]
    fn synapses_weights(&self) -> Vec<i16> {
        self.line.synapses_weights.to_vec()
    }

    #[setter]
    fn set_synapses_weights(&mut self, val: Vec<i16>) {
        for i in 0..32.min(val.len()) {
            self.line.synapses_weights[i] = val[i];
        }
    }

    #[getter]
    fn loopback_address(&self) -> u32 {
        self.line.loopback_address
    }

    #[setter]
    fn set_loopback_address(&mut self, val: u32) {
        self.line.loopback_address = val;
    }

    #[getter]
    fn loopback_energy(&self) -> u8 {
        self.line.loopback_energy
    }

    #[setter]
    fn set_loopback_energy(&mut self, val: u8) {
        self.line.loopback_energy = val;
    }

    #[getter]
    fn local_potential(&self) -> i32 {
        self.line.local_potential
    }

    #[setter]
    fn set_local_potential(&mut self, val: i32) {
        self.line.local_potential = val;
    }

    #[getter]
    fn activation_threshold(&self) -> i32 {
        self.line.activation_threshold
    }

    #[setter]
    fn set_activation_threshold(&mut self, val: i32) {
        self.line.activation_threshold = val;
    }
}

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PySpikingAttentionState {
    pub state: SpikingAttentionState,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySpikingAttentionState {
    #[new]
    fn new() -> Self {
        Self {
            state: SpikingAttentionState::new(),
        }
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn update(&mut self, k: [u32; 8], v: [i16; 8]) {
        self.state.update(&k, &v);
    }

    fn query(&self, q: [u32; 8]) -> [i16; 8] {
        self.state.query(&q)
    }

    fn get_matrix(&self) -> Vec<Vec<i16>> {
        self.state.s.iter().map(|row| row.to_vec()).collect()
    }
}

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyHierarchicalWinnerTakeAll {
    pub wta: HierarchicalWinnerTakeAll,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyHierarchicalWinnerTakeAll {
    #[new]
    fn new() -> Self {
        Self {
            wta: HierarchicalWinnerTakeAll::new(),
        }
    }

    fn reset(&mut self) {
        self.wta.reset();
    }

    #[getter]
    fn macro_potentials(&self) -> Vec<i32> {
        self.wta.macro_potentials.to_vec()
    }

    #[setter]
    fn set_macro_potentials(&mut self, val: Vec<i32>) {
        for i in 0..50.min(val.len()) {
            self.wta.macro_potentials[i] = val[i];
        }
    }

    #[getter]
    fn micro_potentials(&self) -> Vec<i32> {
        self.wta.micro_potentials.clone()
    }

    #[setter]
    fn set_micro_potentials(&mut self, val: Vec<i32>) {
        self.wta.micro_potentials = val;
    }

    fn select_macro_cluster(&self) -> usize {
        self.wta.select_macro_cluster()
    }

    fn select_winning_token(&self, winning_cluster: usize) -> u32 {
        self.wta.select_winning_token(winning_cluster)
    }
}

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyNeuronGuardTrainerField {
    pub trainer: NeuronGuardTrainerField,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyNeuronGuardTrainerField {
    #[new]
    fn new(sensory_count: usize, motor_count: usize) -> Self {
        Self {
            trainer: NeuronGuardTrainerField::new(sensory_count, motor_count),
        }
    }

    fn reset_potentials(&self) {
        self.trainer.reset_potentials();
    }

    fn train_stream_step_sync(&mut self, token_indices: Vec<u32>) {
        self.trainer.train_stream_step_sync(token_indices);
    }

    fn save_weights_to_b64(&self, path: String) -> PyResult<()> {
        let bytes = self.trainer.serialize_weights();
        let b64_str = base64_encode(&bytes);
        std::fs::write(path, b64_str)?;
        Ok(())
    }

    fn load_weights_from_b64(&mut self, path: String) -> PyResult<()> {
        self.trainer.load_weights_from_b64(&path)?;
        Ok(())
    }

    fn process_step_sync(&self, token_indices: Vec<u32>) {
        self.trainer.process_step_sync(token_indices);
    }

    fn decay_potentials(&self, alpha: f32) {
        self.trainer.decay_potentials(alpha);
    }

    fn get_potentials(&self) -> Vec<i32> {
        self.trainer.get_potentials()
    }
}

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyCPGManager;

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyCPGManager {
    #[staticmethod]
    fn propagate_echo(
        line: &PyPermanentNeuromorphicLine,
        mut pool: Vec<PyPermanentNeuromorphicLine>,
    ) -> Vec<PyPermanentNeuromorphicLine> {
        let mut raw_pool: Vec<HighDensityNeuromorphicLine> = pool.iter().map(|p| p.line).collect();
        crate::cpg::propagate_cpg_echo(&line.line, &mut raw_pool);
        for (p, r) in pool.iter_mut().zip(raw_pool.iter()) {
            p.line = *r;
        }
        pool
    }

    #[staticmethod]
    fn decay_potentials(
        mut pool: Vec<PyPermanentNeuromorphicLine>,
        alpha: f32,
    ) -> Vec<PyPermanentNeuromorphicLine> {
        let mut raw_pool: Vec<HighDensityNeuromorphicLine> = pool.iter().map(|p| p.line).collect();
        crate::cpg::decay_potentials(&mut raw_pool, alpha);
        for (p, r) in pool.iter_mut().zip(raw_pool.iter()) {
            p.line = *r;
        }
        pool
    }

    #[staticmethod]
    fn decay_accumulators(mut accumulators: [i32; 256], alpha: f32) -> [i32; 256] {
        crate::cpg::decay_accumulators(&mut accumulators, alpha);
        accumulators
    }
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn neuronguard(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuronGuardField>()?;
    m.add_class::<PyPermanentSpatiotemporalEnsembleMesh>()?;
    m.add_class::<PyInsectoidSimulation>()?;
    m.add_class::<PyPermanentNeuromorphicLine>()?;
    m.add_class::<PySpikingAttentionState>()?;
    m.add_class::<PyHierarchicalWinnerTakeAll>()?;
    m.add_class::<PyCPGManager>()?;
    m.add_class::<PyNeuronGuardTrainerField>()?;
    Ok(())
}
