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

use crate::gen_memory::PermanentNeuromorphicLine;

pub const DV: usize = 8; // Value dimension

/// Evaluates the ternary synaptic weight masking logic (W in {-1, 0, 1}) using binary masks.
/// Performs bitwise AND between input_spikes and synapses_positive/synapses_negative.
/// Bypasses floating-point ALUs entirely by using population count (.count_ones()).
pub fn evaluate_ternary_synapses(line: &PermanentNeuromorphicLine, input_spikes: &[u32; 8]) -> i16 {
    let mut score: i16 = 0;
    for i in 0..8 {
        let pos_active = input_spikes[i] & line.synapses_positive[i];
        let neg_active = input_spikes[i] & line.synapses_negative[i];
        score += pos_active.count_ones() as i16;
        score -= neg_active.count_ones() as i16;
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
    fn test_evaluate_ternary_synapses() {
        let mut line = PermanentNeuromorphicLine {
            synapses_positive: [0; 8],
            synapses_negative: [0; 8],
            loopback_address: 0,
            loopback_energy: 0,
            local_potential: 0,
            activation_threshold: 0,
            _padding: [0; 54],
        };

        // Set some positive and negative synapses
        line.synapses_positive[0] = 0b1011; // Bits 0, 1, 3 are positive
        line.synapses_negative[0] = 0b0100; // Bit 2 is negative

        let mut input_spikes = [0u32; 8];
        input_spikes[0] = 0b1111; // All first 4 bits spike

        let score = evaluate_ternary_synapses(&line, &input_spikes);
        // 3 positive active, 1 negative active -> score should be 3 - 1 = 2
        assert_eq!(score, 2);
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
