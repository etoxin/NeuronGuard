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

use crate::gen_memory::MaxRangeNeuromorphicLine;

/// Propagates CPG recurrent echoes backward to preceding cache locations.
/// If a line's local_potential meets or exceeds its activation_threshold,
/// it echoes loopback_energy backward to the target loopback_address in the memory pool.
pub fn propagate_cpg_echo(
    line: &MaxRangeNeuromorphicLine,
    memory_pool: &mut [MaxRangeNeuromorphicLine],
) {
    if line.local_potential >= line.activation_threshold && line.activation_threshold > 0 {
        let target_idx = line.loopback_address as usize;
        if target_idx < memory_pool.len() {
            memory_pool[target_idx].local_potential = memory_pool[target_idx]
                .local_potential
                .saturating_add(line.loopback_energy as i32);
        }
    }
}

/// Applies a fixed decay factor (alpha) to the local potentials and loopback energies
/// of all neuromorphic lines in the memory pool to prevent saturation.
pub fn decay_potentials(lines: &mut [MaxRangeNeuromorphicLine], alpha: f32) {
    for line in lines.iter_mut() {
        // Decay local potential
        let current_pot = line.local_potential as f32;
        line.local_potential = (current_pot * alpha) as i32;

        // Decay loopback energy
        let current_energy = line.loopback_energy as f32;
        line.loopback_energy = (current_energy * alpha) as u8;
    }
}

/// Applies a fixed decay factor (alpha) to the global memory accumulators.
pub fn decay_accumulators(accumulators: &mut [i32; 256], alpha: f32) {
    for acc in accumulators.iter_mut() {
        let current = *acc as f32;
        *acc = (current * alpha) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propagate_cpg_echo() {
        let mut memory_pool = vec![
            MaxRangeNeuromorphicLine {
                synapses_weights: [0; 32],
                loopback_address: 0,
                loopback_energy: 0,
                local_potential: 10,
                activation_threshold: 15,
                _padding: [0; 51],
            },
            MaxRangeNeuromorphicLine {
                synapses_weights: [0; 32],
                loopback_address: 0, // Echoes back to index 0
                loopback_energy: 5,
                local_potential: 20,
                activation_threshold: 15, // Spikes!
                _padding: [0; 51],
            },
        ];

        // Propagate echo from the spiking line at index 1
        let line_1 = memory_pool[1];
        propagate_cpg_echo(&line_1, &mut memory_pool);

        // Index 0 should have received the loopback energy: 10 + 5 = 15
        assert_eq!(memory_pool[0].local_potential, 15);
    }

    #[test]
    fn test_decay_potentials_and_accumulators() {
        let mut memory_pool = vec![MaxRangeNeuromorphicLine {
            synapses_weights: [0; 32],
            loopback_address: 0,
            loopback_energy: 10,
            local_potential: 100,
            activation_threshold: 15,
            _padding: [0; 51],
        }];

        decay_potentials(&mut memory_pool, 0.90);
        assert_eq!(memory_pool[0].local_potential, 90);
        assert_eq!(memory_pool[0].loopback_energy, 9);

        let mut accumulators = [100i32; 256];
        decay_accumulators(&mut accumulators, 0.50);
        assert_eq!(accumulators[0], 50);
        assert_eq!(accumulators[255], 50);
    }
}
