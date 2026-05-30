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
use std::sync::atomic::{AtomicI16, Ordering};

/// NeuronGuardTrainerField
/// Manages the pre-allocated, 128-byte aligned neuromorphic memory matrix for training.
pub struct NeuronGuardTrainerField {
    pub sensory_count: usize,
    pub motor_count: usize,
    pub lines: Vec<PermanentNeuromorphicLine>,
    pub potentials: Vec<AtomicI16>,
}

impl NeuronGuardTrainerField {
    /// Allocates a flat, contiguous block of memory for sensory and motor neurons.
    pub fn new(sensory_count: usize, motor_count: usize) -> Self {
        let mut lines = Vec::with_capacity(sensory_count);
        for _ in 0..sensory_count {
            lines.push(PermanentNeuromorphicLine {
                synapses_positive: [0; 8],
                synapses_negative: [0; 8],
                loopback_address: 0,
                loopback_energy: 0,
                local_potential: 0,
                activation_threshold: 15,
                _padding: [0; 54],
            });
        }

        let mut potentials = Vec::with_capacity(motor_count);
        for _ in 0..motor_count {
            potentials.push(AtomicI16::new(0));
        }

        Self {
            sensory_count,
            motor_count,
            lines,
            potentials,
        }
    }

    /// Resets all potentials to zero.
    pub fn reset_potentials(&self) {
        for pot in &self.potentials {
            pot.store(0, Ordering::Relaxed);
        }
    }

    /// Executes a single-pass Spike-Driven Hebbian Plasticity training step on a stream of token indices.
    /// Completely bypasses the processor's floating-point ALUs.
    pub fn train_stream_step_sync(&mut self, token_indices: Vec<u32>) {
        if token_indices.len() < 2 {
            return;
        }

        for t in 0..token_indices.len() - 1 {
            let xt = token_indices[t] as usize;
            let xt_next = token_indices[t + 1] as usize;

            if xt >= self.sensory_count || xt_next >= self.motor_count {
                continue;
            }

            // 1. Sensory Injection & Potentials Accumulation
            let line = &self.lines[xt];
            for j in 0..256 {
                let word_idx = j >> 5;
                let bit_idx = j & 31;
                let target_token_id = (xt + j) % self.motor_count;

                // Check positive synapses
                if (line.synapses_positive[word_idx] & (1 << bit_idx)) != 0 {
                    self.potentials[target_token_id].fetch_add(1, Ordering::Relaxed);
                }
                // Check negative synapses
                if (line.synapses_negative[word_idx] & (1 << bit_idx)) != 0 {
                    self.potentials[target_token_id].fetch_sub(1, Ordering::Relaxed);
                }
            }

            // 2. Local Error Evaluation
            let mut prediction = 0;
            let mut max_potential = i16::MIN;
            for i in 0..self.motor_count {
                let pot = self.potentials[i].load(Ordering::Relaxed);
                if pot > max_potential {
                    max_potential = pot;
                    prediction = i;
                }
            }

            // 3. Synaptic Update (Hebbian Rule)
            // Potentiation: reinforce connection to xt_next
            let j_next = (xt_next + self.motor_count - xt) % self.motor_count;
            if j_next < 256 {
                let word_idx = j_next >> 5;
                let bit_idx = j_next & 31;
                self.lines[xt].synapses_positive[word_idx] |= 1 << bit_idx;
                self.lines[xt].synapses_negative[word_idx] &= !(1 << bit_idx);
            }

            // Depression: penalize connection to incorrect prediction
            if prediction != xt_next {
                let k_pred = (prediction + self.motor_count - xt) % self.motor_count;
                if k_pred < 256 {
                    let word_idx = k_pred >> 5;
                    let bit_idx = k_pred & 31;
                    self.lines[xt].synapses_negative[word_idx] |= 1 << bit_idx;
                    self.lines[xt].synapses_positive[word_idx] &= !(1 << bit_idx);
                }
            }

            // 4. Decay Step (alpha = 0.90)
            for pot in &self.potentials {
                let current = pot.load(Ordering::Relaxed);
                if current != 0 {
                    let decayed = (current as f32 * 0.90) as i16;
                    pot.store(decayed, Ordering::Relaxed);
                }
            }
        }
    }

    /// Serializes the final synaptic matrix directly into a flat, contiguous binary array.
    pub fn serialize_weights(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.sensory_count * 64);
        for line in &self.lines {
            for &val in &line.synapses_positive {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            for &val in &line.synapses_negative {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }
        bytes
    }

    /// Loads and deserializes the synaptic matrix from a base64-encoded text file.
    pub fn load_weights_from_b64(&mut self, path: &str) -> std::io::Result<()> {
        let b64_str = std::fs::read_to_string(path)?;
        let bytes = base64_decode(&b64_str);

        let mut offset = 0;
        for line in &mut self.lines {
            if offset + 64 > bytes.len() {
                break;
            }
            for j in 0..8 {
                line.synapses_positive[j] =
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            for j in 0..8 {
                line.synapses_negative[j] =
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
        }
        Ok(())
    }

    /// Processes a stream of token indices synchronously for inference.
    pub fn process_step_sync(&self, token_indices: Vec<u32>) {
        for &xt in &token_indices {
            let xt = xt as usize;
            if xt >= self.sensory_count {
                continue;
            }
            let line = &self.lines[xt];
            for j in 0..256 {
                let word_idx = j >> 5;
                let bit_idx = j & 31;
                let target_token_id = (xt + j) % self.motor_count;

                if (line.synapses_positive[word_idx] & (1 << bit_idx)) != 0 {
                    self.potentials[target_token_id].fetch_add(1, Ordering::Relaxed);
                }
                if (line.synapses_negative[word_idx] & (1 << bit_idx)) != 0 {
                    self.potentials[target_token_id].fetch_sub(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Applies a fixed decay factor to all potentials.
    pub fn decay_potentials(&self, alpha: f32) {
        for pot in &self.potentials {
            let current = pot.load(Ordering::Relaxed);
            if current != 0 {
                let decayed = (current as f32 * alpha) as i16;
                pot.store(decayed, Ordering::Relaxed);
            }
        }
    }

    /// Returns the current potentials of all motor neurons.
    pub fn get_potentials(&self) -> Vec<i16> {
        self.potentials
            .iter()
            .map(|pot| pot.load(Ordering::Relaxed))
            .collect()
    }
}

/// High-performance, zero-dependency Base64 encoder.
pub fn base64_encode(bytes: &[u8]) -> String {
    const CHARSET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        match chunk.len() {
            3 => {
                let b0 = chunk[0] as usize;
                let b1 = chunk[1] as usize;
                let b2 = chunk[2] as usize;
                result.push(CHARSET[b0 >> 2] as char);
                result.push(CHARSET[((b0 & 3) << 4) | (b1 >> 4)] as char);
                result.push(CHARSET[((b1 & 15) << 2) | (b2 >> 6)] as char);
                result.push(CHARSET[b2 & 63] as char);
            }
            2 => {
                let b0 = chunk[0] as usize;
                let b1 = chunk[1] as usize;
                result.push(CHARSET[b0 >> 2] as char);
                result.push(CHARSET[((b0 & 3) << 4) | (b1 >> 4)] as char);
                result.push(CHARSET[(b1 & 15) << 2] as char);
                result.push('=');
            }
            1 => {
                let b0 = chunk[0] as usize;
                result.push(CHARSET[b0 >> 2] as char);
                result.push(CHARSET[(b0 & 3) << 4] as char);
                result.push('=');
                result.push('=');
            }
            _ => unreachable!(),
        }
    }
    result
}

/// High-performance, zero-dependency Base64 decoder.
pub fn base64_decode(b64_str: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0;
    for c in b64_str.chars() {
        let val = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            '=' => continue,
            _ => continue, // Ignore whitespace/newlines
        };
        buffer = (buffer << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((buffer >> bits) as u8);
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_field_creation_and_reset() {
        let trainer = NeuronGuardTrainerField::new(100, 100);
        assert_eq!(trainer.sensory_count, 100);
        assert_eq!(trainer.motor_count, 100);
        assert_eq!(trainer.lines.len(), 100);
        assert_eq!(trainer.potentials.len(), 100);

        trainer.potentials[42].store(10, Ordering::Relaxed);
        trainer.reset_potentials();
        assert_eq!(trainer.potentials[42].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_hebbian_training_step() {
        let mut trainer = NeuronGuardTrainerField::new(10, 10);

        // Train on sequence: 2 -> 5
        trainer.train_stream_step_sync(vec![2, 5]);

        // Synaptic pathway from 2 to 5 should be potentiated
        let j_next = (5 + 10 - 2) % 10; // 3
        let word_idx = j_next >> 5;
        let bit_idx = j_next & 31;
        assert_ne!(
            trainer.lines[2].synapses_positive[word_idx] & (1 << bit_idx),
            0
        );
    }

    #[test]
    fn test_base64_encode() {
        let data = b"hello";
        let encoded = base64_encode(data);
        assert_eq!(encoded, "aGVsbG8=");
    }
}
