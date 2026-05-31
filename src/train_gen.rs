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
use std::io::{Read, Write};
use std::sync::atomic::{AtomicI32, Ordering};

/// NeuronGuardTrainerField
/// Manages the pre-allocated, 128-byte aligned neuromorphic memory matrix for training.
pub struct NeuronGuardTrainerField {
    pub sensory_count: usize,
    pub motor_count: usize,
    pub lines: Vec<HighDensityNeuromorphicLine>,
    pub potentials: Vec<AtomicI32>,
    pub macro_potentials: Vec<AtomicI32>, // Added for fast hierarchical WTA search
}

impl NeuronGuardTrainerField {
    /// Allocates a flat, contiguous block of memory for sensory and motor neurons.
    pub fn new(sensory_count: usize, motor_count: usize) -> Self {
        let mut lines = Vec::with_capacity(sensory_count);
        for _ in 0..sensory_count {
            lines.push(HighDensityNeuromorphicLine::new(15));
        }

        let mut potentials = Vec::with_capacity(motor_count);
        for _ in 0..motor_count {
            potentials.push(AtomicI32::new(0));
        }

        // 50 macro clusters of size 1000
        let num_macro = (motor_count + 999) / 1000;
        let mut macro_potentials = Vec::with_capacity(num_macro);
        for _ in 0..num_macro {
            macro_potentials.push(AtomicI32::new(0));
        }

        Self {
            sensory_count,
            motor_count,
            lines,
            potentials,
            macro_potentials,
        }
    }

    /// Resets all potentials to zero.
    pub fn reset_potentials(&self) {
        for pot in &self.potentials {
            pot.store(0, Ordering::Relaxed);
        }
        for m_pot in &self.macro_potentials {
            m_pot.store(0, Ordering::Relaxed);
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
            for j in 0..56 {
                let target_token_id = (xt + j) % self.motor_count;
                let macro_idx = target_token_id / 1000;
                let weight = line.synapses_weights[j] as i32;

                if weight != 0 {
                    self.potentials[target_token_id].fetch_add(weight, Ordering::Relaxed);
                    if macro_idx < self.macro_potentials.len() {
                        self.macro_potentials[macro_idx].fetch_add(weight, Ordering::Relaxed);
                    }
                }
            }

            // 2. Local Error Evaluation (Hierarchical WTA Search: O(1) cache-resident)
            // Tier-1: Find winning macro cluster (50 elements)
            let mut winning_cluster = 0;
            let mut max_macro_pot = i32::MIN;
            for i in 0..self.macro_potentials.len() {
                let pot = self.macro_potentials[i].load(Ordering::Relaxed);
                if pot > max_macro_pot {
                    max_macro_pot = pot;
                    winning_cluster = i;
                }
            }

            // Tier-2: Find winning token within the winning cluster (1,000 elements)
            let start_idx = winning_cluster * 1000;
            let end_idx = (start_idx + 1000).min(self.motor_count);
            let mut prediction = 0;
            let mut max_potential = i32::MIN;
            for i in start_idx..end_idx {
                let pot = self.potentials[i].load(Ordering::Relaxed);
                if pot > max_potential {
                    max_potential = pot;
                    prediction = i;
                }
            }

            // 3. Synaptic Update (Hebbian Rule)
            // Potentiation: reinforce connection to xt_next
            let j_next = (xt_next + self.motor_count - xt) % self.motor_count;
            if j_next < 56 {
                self.lines[xt].adjust_synapse(j_next, 100); // Upgraded from 1 to 100 for stronger associations
            }

            // Depression: penalize connection to incorrect prediction
            if prediction != xt_next {
                let k_pred = (prediction + self.motor_count - xt) % self.motor_count;
                if k_pred < 56 {
                    self.lines[xt].adjust_synapse(k_pred, -50); // Upgraded from 1 to 50 for stronger penalty
                }
            }

            // 4. Decay Step (applied once every 100 steps to keep the hot path O(1))
            if t % 100 == 0 {
                for pot in &self.potentials {
                    let current = pot.load(Ordering::Relaxed);
                    if current != 0 {
                        let decayed = (current as f32 * 0.90) as i32;
                        pot.store(decayed, Ordering::Relaxed);
                    }
                }
                for m_pot in &self.macro_potentials {
                    let current = m_pot.load(Ordering::Relaxed);
                    if current != 0 {
                        let decayed = (current as f32 * 0.90) as i32;
                        m_pot.store(decayed, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    /// Serializes the final synaptic matrix directly into a flat, contiguous binary array.
    pub fn serialize_weights(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.sensory_count * 112);
        for line in &self.lines {
            for &val in &line.synapses_weights {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }
        bytes
    }

    /// Serializes and writes the synaptic matrix directly to a base64-encoded text file in chunks.
    /// This prevents memory spikes and string buffer overflows on large matrices (up to 2 GB).
    pub fn save_weights_to_b64(&self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Buffer to hold raw bytes for a chunk of lines
        let chunk_size = 1002; // 1,002 lines * 112 bytes = 112,224 bytes (multiple of 3 for perfect base64 alignment)
        let mut chunk_bytes = Vec::with_capacity(chunk_size * 112);

        for line in &self.lines {
            for &val in &line.synapses_weights {
                chunk_bytes.extend_from_slice(&val.to_le_bytes());
            }

            if chunk_bytes.len() >= chunk_size * 112 {
                let b64_str = base64_encode(&chunk_bytes);
                file.write_all(b64_str.as_bytes())?;
                chunk_bytes.clear();
            }
        }

        if !chunk_bytes.is_empty() {
            let b64_str = base64_encode(&chunk_bytes);
            file.write_all(b64_str.as_bytes())?;
        }

        Ok(())
    }

    /// Loads and deserializes the synaptic matrix from a base64-encoded text file in chunks.
    /// This prevents memory spikes and string buffer overflows on large matrices (up to 2 GB).
    pub fn load_weights_from_b64(&mut self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::open(path)?;

        // We read the file in chunks of 149,632 base64 characters (which decodes to exactly 1,002 lines)
        let b64_chunk_size = 149_632;
        let mut b64_buffer = vec![0u8; b64_chunk_size];

        let mut line_idx = 0;
        loop {
            let bytes_read = read_exact_or_eof(&mut file, &mut b64_buffer)?;
            if bytes_read == 0 {
                break;
            }

            let b64_str = std::str::from_utf8(&b64_buffer[..bytes_read])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let bytes = base64_decode(b64_str);

            let mut offset = 0;
            while offset + 112 <= bytes.len() && line_idx < self.lines.len() {
                let line = &mut self.lines[line_idx];
                for j in 0..56 {
                    line.synapses_weights[j] =
                        i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                    offset += 2;
                }
                line_idx += 1;
            }
        }

        if line_idx != self.lines.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Synaptic weights file size mismatch: expected {} lines, loaded {} lines",
                    self.lines.len(),
                    line_idx
                ),
            ));
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
            for j in 0..56 {
                let target_token_id = (xt + j) % self.motor_count;
                let weight = line.synapses_weights[j] as i32;
                if weight != 0 {
                    self.potentials[target_token_id].fetch_add(weight, Ordering::Relaxed);
                }
            }
        }
    }

    /// Applies a fixed decay factor to all potentials.
    pub fn decay_potentials(&self, alpha: f32) {
        for pot in &self.potentials {
            let current = pot.load(Ordering::Relaxed);
            if current != 0 {
                let decayed = (current as f32 * alpha) as i32;
                pot.store(decayed, Ordering::Relaxed);
            }
        }
    }

    /// Returns the current potentials of all motor neurons.
    pub fn get_potentials(&self) -> Vec<i32> {
        self.potentials
            .iter()
            .map(|pot| pot.load(Ordering::Relaxed))
            .collect()
    }
}

/// Helper function to read exactly `buf.len()` bytes or stop at EOF.
fn read_exact_or_eof(file: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut total_read = 0;
    while total_read < buf.len() {
        let bytes_read = file.read(&mut buf[total_read..])?;
        if bytes_read == 0 {
            break;
        }
        total_read += bytes_read;
    }
    Ok(total_read)
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
        assert_eq!(trainer.lines[2].synapses_weights[j_next], 100);
    }

    #[test]
    fn test_base64_encode() {
        let data = b"hello";
        let encoded = base64_encode(data);
        assert_eq!(encoded, "aGVsbG8=");
    }
}
