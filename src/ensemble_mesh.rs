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

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::Mutex;

pub const MAX_FORWARD_SLOTS: usize = 6;
pub const MAX_BACKWARD_SLOTS: usize = 4;

/// PermanentNeuromorphicLine
/// Spatially aligned to exactly 64 bytes to fill a standard CPU cache line.
/// Bypasses traditional deep-learning bottlenecks with a lean, matrix-free architecture.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct PermanentNeuromorphicLine {
    pub token_id: u32,                               // 4 Bytes: Unique structural index
    pub active_forward: u32,                         // 4 Bytes: Bounds checked count (<= 6)
    pub active_backward: u32,                        // 4 Bytes: Bounds checked count (<= 4)
    pub forward_targets: [u32; MAX_FORWARD_SLOTS],   // 24 Bytes: Next-state target indexes
    pub backward_targets: [u32; MAX_BACKWARD_SLOTS], // 16 Bytes: Loopback context target indexes
    pub padding: [u8; 12],                           // 12 Bytes: Strict cache-line padding
}

impl PermanentNeuromorphicLine {
    /// Updates the recurrent loopback targets with the recent history of active tokens.
    pub fn update_backward_targets(&mut self, history: &[u32]) {
        let count = history.len().min(MAX_BACKWARD_SLOTS);
        for i in 0..count {
            self.backward_targets[i] = history[i];
        }
        self.active_backward = count as u32;
    }

    /// Updates an existing forward target or evicts the weakest one if capacity is reached.
    /// The weakest target is defined as the one with the lowest accumulator value.
    pub fn update_or_evict_forward_target(
        &mut self,
        correct_target: u32,
        accumulators: &[AtomicI32],
    ) -> Option<u32> {
        // Check if the target already exists
        for i in 0..self.active_forward as usize {
            if self.forward_targets[i] == correct_target {
                return None;
            }
        }

        if (self.active_forward as usize) < MAX_FORWARD_SLOTS {
            let idx = self.active_forward as usize;
            self.forward_targets[idx] = correct_target;
            self.active_forward += 1;
            None
        } else {
            // Evict the weakest target (lowest accumulator value)
            let mut weakest_idx = 0;
            let mut weakest_val = i32::MAX;

            for i in 0..MAX_FORWARD_SLOTS {
                let target = self.forward_targets[i] as usize;
                let val = if target < accumulators.len() {
                    accumulators[target].load(Ordering::Relaxed)
                } else {
                    0
                };
                if val < weakest_val {
                    weakest_val = val;
                    weakest_idx = i;
                }
            }

            let evicted = self.forward_targets[weakest_idx];
            self.forward_targets[weakest_idx] = correct_target;
            Some(evicted)
        }
    }
}

/// StructuralMutationEvent
/// Emitted when a Guard/Lease operation executes a hard structural swap.
#[derive(Debug, Clone, Copy)]
pub struct StructuralMutationEvent {
    pub token_id: u32,
    pub evicted_target: u32,
    pub new_target: u32,
    pub timestamp_us: u64,
}

/// PermanentSpatiotemporalEnsembleMesh
/// A flat, contiguous block of memory for PermanentNeuromorphicLines.
pub struct PermanentSpatiotemporalEnsembleMesh {
    pub storage: *mut PermanentNeuromorphicLine,
    pub size: usize,
    pub size_per_field: usize,
    pub global_accumulators: Vec<AtomicI32>,
    pub active_history: Mutex<Vec<u32>>,
    pub mutation_events: Mutex<Vec<StructuralMutationEvent>>,
}

impl PermanentSpatiotemporalEnsembleMesh {
    /// Allocates a flat, contiguous block of memory for `size_per_field * 3` lines, zero-initialized.
    pub fn new(size_per_field: usize) -> Self {
        let size = size_per_field * 3;
        let layout = Layout::array::<PermanentNeuromorphicLine>(size)
            .expect("Failed to create memory layout for PermanentSpatiotemporalEnsembleMesh");

        let storage = unsafe {
            let ptr = alloc_zeroed(layout) as *mut PermanentNeuromorphicLine;
            if ptr.is_null() {
                panic!("Failed to allocate memory for PermanentSpatiotemporalEnsembleMesh");
            }
            ptr
        };

        // Initialize token_ids
        unsafe {
            for i in 0..size {
                (*storage.add(i)).token_id = i as u32;
            }
        }

        let mut global_accumulators = Vec::with_capacity(size);
        for _ in 0..size {
            global_accumulators.push(AtomicI32::new(0));
        }

        Self {
            storage,
            size,
            size_per_field,
            global_accumulators,
            active_history: Mutex::new(Vec::new()),
            mutation_events: Mutex::new(Vec::new()),
        }
    }

    /// Pure pointerless offset arithmetic mapping to Base + ID * 64
    ///
    /// # Safety
    /// Raw pointer arithmetic. Caller must ensure `id` is within bounds.
    pub unsafe fn get_line(&self, id: usize) -> &mut PermanentNeuromorphicLine {
        if id >= self.size {
            panic!("Line ID out of bounds: {} >= {}", id, self.size);
        }
        &mut *self.storage.add(id)
    }

    /// Acquires a transactional, lock-free lease on the specific 64-byte memory address of the active token.
    /// Uses the first 4 bytes of padding as an AtomicU32 lease flag.
    pub fn try_acquire_lease(&self, id: usize) -> Option<PermanentLineLease<'_>> {
        if id >= self.size {
            return None;
        }
        unsafe {
            let line = self.get_line(id);
            let lease_ptr = line.padding.as_ptr() as *const AtomicU32;
            let lease_ref = &*lease_ptr;
            if lease_ref
                .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                Some(PermanentLineLease {
                    line_id: id,
                    mesh: self,
                })
            } else {
                None
            }
        }
    }

    /// Processes a single frame pass by evaluating 3 concurrent token structures.
    pub fn process_frame(
        &self,
        tokens: [u32; 3],
        training_mode: bool,
        correct_target: Option<u32>,
        decay_factor: f32,
    ) {
        for &token_id in &tokens {
            let token_idx = token_id as usize;
            if token_idx >= self.size {
                continue;
            }

            unsafe {
                let line = self.get_line(token_idx);

                // Activate forward targets
                for i in 0..line.active_forward as usize {
                    let target = line.forward_targets[i] as usize;
                    if target < self.global_accumulators.len() {
                        self.global_accumulators[target].fetch_add(1, Ordering::Relaxed);
                    }
                }

                // Activate backward targets
                for i in 0..line.active_backward as usize {
                    let target = line.backward_targets[i] as usize;
                    if target < self.global_accumulators.len() {
                        self.global_accumulators[target].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            // Update recurrent loopback targets for this token
            {
                let mut history = self.active_history.lock().unwrap();
                if let Some(lease) = self.try_acquire_lease(token_idx) {
                    let line = lease.line();
                    line.update_backward_targets(&history);
                }

                // Push current token to history
                history.insert(0, token_id);
                if history.len() > MAX_BACKWARD_SLOTS {
                    history.truncate(MAX_BACKWARD_SLOTS);
                }
            }

            // If training mode and correct target is provided, perform topological plasticity
            if training_mode {
                if let Some(target) = correct_target {
                    if let Some(lease) = self.try_acquire_lease(token_idx) {
                        let line = lease.line();
                        if let Some(evicted) =
                            line.update_or_evict_forward_target(target, &self.global_accumulators)
                        {
                            // Emit structural mutation event
                            let mut events = self.mutation_events.lock().unwrap();
                            events.push(StructuralMutationEvent {
                                token_id,
                                evicted_target: evicted,
                                new_target: target,
                                timestamp_us: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_micros()
                                    as u64,
                            });
                        }
                    }
                }
            }
        }

        // Apply background decay
        self.decay(decay_factor);
    }

    /// Shaves a percentage of integer value off the accumulators on every processing cycle.
    pub fn decay(&self, decay_factor: f32) {
        for acc in &self.global_accumulators {
            let current = acc.load(Ordering::Relaxed);
            if current != 0 {
                let decayed = (current as f32 * decay_factor) as i32;
                acc.store(decayed, Ordering::Relaxed);
            }
        }
    }

    // --- Telemetry Interfaces ---

    /// Returns an array of currently active token IDs to flash corresponding 3D graph nodes.
    pub fn get_active_nodes(&self) -> Vec<u32> {
        let mut active = Vec::new();
        for i in 0..self.size {
            if self.global_accumulators[i].load(Ordering::Relaxed) > 0 {
                active.push(i as u32);
            }
        }
        active
    }

    /// Returns a map of current values from the global_accumulators array.
    pub fn get_loop_intensities(&self) -> HashMap<u32, i32> {
        let mut intensities = HashMap::new();
        for i in 0..self.size {
            let val = self.global_accumulators[i].load(Ordering::Relaxed);
            if val > 0 {
                intensities.insert(i as u32, val);
            }
        }
        intensities
    }

    /// Emits and drains structural mutation events.
    pub fn get_structural_mutations(&self) -> Vec<StructuralMutationEvent> {
        let mut events = self.mutation_events.lock().unwrap();
        std::mem::take(&mut *events)
    }
}

impl Drop for PermanentSpatiotemporalEnsembleMesh {
    fn drop(&mut self) {
        let layout = Layout::array::<PermanentNeuromorphicLine>(self.size)
            .expect("Failed to create layout for deallocation");
        unsafe {
            dealloc(self.storage as *mut u8, layout);
        }
    }
}

unsafe impl Send for PermanentSpatiotemporalEnsembleMesh {}
unsafe impl Sync for PermanentSpatiotemporalEnsembleMesh {}

/// PermanentLineLease
/// Represents a transactional lease on a specific permanent neuromorphic line.
pub struct PermanentLineLease<'a> {
    pub line_id: usize,
    pub mesh: &'a PermanentSpatiotemporalEnsembleMesh,
}

impl<'a> PermanentLineLease<'a> {
    pub fn line(&self) -> &mut PermanentNeuromorphicLine {
        unsafe { self.mesh.get_line(self.line_id) }
    }
}

impl<'a> Drop for PermanentLineLease<'a> {
    fn drop(&mut self) {
        unsafe {
            let line = self.mesh.get_line(self.line_id);
            let lease_ptr = line.padding.as_ptr() as *const AtomicU32;
            let lease_ref = &*lease_ptr;
            lease_ref.store(0, Ordering::SeqCst);
        }
    }
}

// --- Insectoid Rigid-Body Walking Gait Simulation ---

pub struct InsectoidSimulation {
    pub mesh: PermanentSpatiotemporalEnsembleMesh,
    pub phases: [f32; 6],
    pub velocities: [f32; 6],
}

impl InsectoidSimulation {
    pub fn new() -> Self {
        let mesh = PermanentSpatiotemporalEnsembleMesh::new(100);

        // Set up sequential forward and backward targets for walking gait: 0 -> 2 -> 4 -> 1 -> 3 -> 5 -> 0
        let sequence = [0, 2, 4, 1, 3, 5];
        for i in 0..6 {
            let current = sequence[i];
            let next = sequence[(i + 1) % 6];
            let prev = sequence[(i + 5) % 6];

            unsafe {
                let line = mesh.get_line(current);
                line.forward_targets[0] = next as u32;
                line.active_forward = 1;

                line.backward_targets[0] = prev as u32;
                line.active_backward = 1;
            }
        }

        Self {
            mesh,
            phases: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], // Initial out-of-sync phases
            velocities: [0.1; 6],
        }
    }

    pub fn step(&mut self, training_mode: bool, correct_target: Option<u32>, decay_factor: f32) {
        // 1. Tokenize current state into 3 parallel semantic fields
        let tokens = tokenize_insectoid_state(&self.phases, &self.velocities);

        // 2. Process frame in the neuromorphic mesh
        self.mesh
            .process_frame(tokens, training_mode, correct_target, decay_factor);

        // 3. Update phases based on accumulator energy
        for i in 0..6 {
            let energy = self.mesh.global_accumulators[i].load(Ordering::Relaxed) as f32;
            self.velocities[i] = 0.05 + 0.01 * energy;
            self.phases[i] += self.velocities[i];
            if self.phases[i] > 2.0 * std::f32::consts::PI {
                self.phases[i] -= 2.0 * std::f32::consts::PI;
            }
        }
    }
}

/// Tokenizes the continuous insectoid state into 3 parallel semantic fields.
pub fn tokenize_insectoid_state(phases: &[f32; 6], velocities: &[f32; 6]) -> [u32; 3] {
    // 1. Spatial Position: Find the leg with the maximum phase
    let mut max_phase = -1.0;
    let mut spatial_leg = 0;
    for i in 0..6 {
        if phases[i] > max_phase {
            max_phase = phases[i];
            spatial_leg = i;
        }
    }
    let spatial_token = spatial_leg as u32; // 0..5

    // 2. Kinetic Velocity: Average velocity quantized to 3 levels
    let avg_vel: f32 = velocities.iter().sum::<f32>() / 6.0;
    let vel_bucket = if avg_vel > 0.15 {
        2
    } else if avg_vel > 0.05 {
        1
    } else {
        0
    };
    let velocity_token = 100 + vel_bucket; // 100..102

    // 3. Phase Delta: Coordination state (Tripod A vs Tripod B phase difference)
    let avg_a = (phases[0] + phases[2] + phases[4]) / 3.0;
    let avg_b = (phases[1] + phases[3] + phases[5]) / 3.0;
    let diff = (avg_a - avg_b).abs();
    let diff_bucket = if diff > 2.0 {
        2
    } else if diff > 1.0 {
        1
    } else {
        0
    };
    let phase_delta_token = 200 + diff_bucket; // 200..202

    [spatial_token, velocity_token, phase_delta_token]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_permanent_line_size_and_alignment() {
        assert_eq!(size_of::<PermanentNeuromorphicLine>(), 64);
        assert_eq!(align_of::<PermanentNeuromorphicLine>(), 64);
    }

    #[test]
    fn test_mesh_allocation_and_lease() {
        let mesh = PermanentSpatiotemporalEnsembleMesh::new(10);
        assert_eq!(mesh.size, 30);

        let lease1 = mesh.try_acquire_lease(5);
        assert!(lease1.is_some());

        let lease2 = mesh.try_acquire_lease(5);
        assert!(lease2.is_none());

        drop(lease1);

        let lease3 = mesh.try_acquire_lease(5);
        assert!(lease3.is_some());
    }

    #[test]
    fn test_walking_gait_stability() {
        let mut sim = InsectoidSimulation::new();

        // Run the simulation for 100 steps to establish a steady walking gait
        for _ in 0..100 {
            sim.step(false, None, 0.90);
        }

        // Verify that the average velocity is stable and non-zero
        let avg_vel: f32 = sim.velocities.iter().sum::<f32>() / 6.0;
        assert!(avg_vel > 0.01 && avg_vel < 1.0);

        // Verify that the accumulators are bounded and don't saturate to infinity
        for i in 0..6 {
            let val = sim.mesh.global_accumulators[i].load(Ordering::Relaxed);
            assert!(val < 100);
        }
    }

    #[test]
    fn test_perturbation_and_repatch() {
        let mut sim = InsectoidSimulation::new();

        // 1. Run to establish steady gait
        for _ in 0..50 {
            sim.step(false, None, 0.90);
        }

        // Initialize token 99 to be full of forward targets to force an eviction
        unsafe {
            let line = sim.mesh.get_line(99);
            line.forward_targets = [10, 11, 12, 13, 14, 15];
            line.active_forward = 6;
        }

        // 2. Inject perturbation (anomalous token 99) and measure re-patch time
        let start_time = std::time::Instant::now();

        // Trigger transactional structural re-patch of token 99 to target leg 0
        sim.mesh.process_frame([99, 101, 201], true, Some(0), 0.90);

        let elapsed = start_time.elapsed();
        // Ensure the transactional structural re-patch mechanism completes within 16.6ms
        assert!(
            elapsed.as_millis() < 16,
            "Re-patch took too long: {:?}",
            elapsed
        );

        // Verify that structural mutation event was emitted
        let mutations = sim.mesh.get_structural_mutations();
        assert!(!mutations.is_empty());
        assert_eq!(mutations[0].token_id, 99);
        assert_eq!(mutations[0].new_target, 0);

        // 3. Continue simulation to ensure self-stabilization
        for _ in 0..50 {
            sim.step(false, None, 0.90);
        }

        let avg_vel: f32 = sim.velocities.iter().sum::<f32>() / 6.0;
        assert!(avg_vel > 0.01 && avg_vel < 1.0);
    }
}
