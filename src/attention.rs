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

use crate::gen_memory::HighDensityNeuromorphicLine;

pub const DV: usize = 8; // Value dimension

/// Evaluates high-resolution synaptic weights.
/// Performs direct multiplication and saturating addition without any bitwise unpacking.
pub fn evaluate_high_res_synapses(
    line: &HighDensityNeuromorphicLine,
    input_spikes: &[i16; 24],
) -> i32 {
    let mut score: i32 = 0;
    for i in 0..24 {
        score = score.saturating_add(line.synapses_weights[i] as i32 * input_spikes[i] as i32);
    }
    score
}

/// SpikingAttentionState
/// Manages the pre-allocated, stack-resident attention state matrix S of size 256 x DV.
/// This guarantees zero heap allocations in the inference path.
#[derive(Debug, Clone, Copy)]
pub struct SpikingAttentionState {
    pub s: [[i16; DV]; 256],
}

impl SpikingAttentionState {
    /// Creates a new zero-initialized attention state.
    pub fn new() -> Self {
        Self { s: [[0; DV]; 256] }
    }

    /// Resets the attention state to zero.
    pub fn reset(&mut self) {
        self.s = [[0; DV]; 256];
    }

    /// Updates the state matrix: S_t = S_{t-1} + (K_t^T x V_t)
    /// K_t is represented as a 256-bit binary spike pattern [u32; 8].
    /// V_t is represented as [i16; DV].
    /// This operates as a sparse array of integer additions, skipping floating-point ALUs.
    pub fn update(&mut self, k: &[u32; 8], v: &[i16; DV]) {
        for word_idx in 0..8 {
            let mut mask = k[word_idx];
            while mask != 0 {
                // Get index of the lowest set bit
                let bit_idx = mask.trailing_zeros() as usize;
                let global_bit_idx = (word_idx << 5) + bit_idx;

                // Add V_t to the corresponding row of S
                for m in 0..DV {
                    self.s[global_bit_idx][m] = self.s[global_bit_idx][m].saturating_add(v[m]);
                }

                // Clear the lowest set bit
                mask &= mask - 1;
            }
        }
    }

    /// Queries the state matrix: Attention(Q, K, V) = Q x S_t
    /// Q is represented as a 256-bit binary spike pattern [u32; 8].
    /// Returns the resulting vector [i16; DV].
    pub fn query(&self, q: &[u32; 8]) -> [i16; DV] {
        let mut result = [0i16; DV];
        for word_idx in 0..8 {
            let mut mask = q[word_idx];
            while mask != 0 {
                let bit_idx = mask.trailing_zeros() as usize;
                let global_bit_idx = (word_idx << 5) + bit_idx;

                for m in 0..DV {
                    result[m] = result[m].saturating_add(self.s[global_bit_idx][m]);
                }

                mask &= mask - 1;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_high_res_synapses() {
        let mut line = HighDensityNeuromorphicLine::new(2); // ADAPTIVE FIX: Lowered from 15 to 2

        // Set some synaptic weights
        line.synapses_weights[0] = 100;
        line.synapses_weights[1] = -50;
        line.synapses_weights[2] = 200;

        let mut input_spikes = [0i16; 24];
        input_spikes[0] = 1;
        input_spikes[1] = 2;
        input_spikes[2] = 1;

        let score = evaluate_high_res_synapses(&line, &input_spikes);
        // 100*1 + (-50)*2 + 200*1 = 100 - 100 + 200 = 200
        assert_eq!(score, 200);
    }

    #[test]
    fn test_spiking_linear_attention() {
        let mut state = SpikingAttentionState::new();

        // Key spike pattern: bits 5 and 42 are active
        let mut k = [0u32; 8];
        k[0] |= 1 << 5;
        k[1] |= 1 << (42 - 32);

        // Value vector
        let v = [1, -2, 3, -4, 5, -6, 7, -8];

        // Update attention state
        state.update(&k, &v);

        // Query spike pattern: bit 5 is active
        let mut q = [0u32; 8];
        q[0] |= 1 << 5;

        let result = state.query(&q);
        assert_eq!(result, v);

        // Query spike pattern: both bit 5 and 42 are active
        let mut q2 = [0u32; 8];
        q2[0] |= 1 << 5;
        q2[1] |= 1 << (42 - 32);

        let result2 = state.query(&q2);
        let expected2 = [2, -4, 6, -8, 10, -12, 14, -16];
        assert_eq!(result2, expected2);
    }
}
