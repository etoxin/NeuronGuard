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

use crate::gen_memory::{AdjustResult, HighDensityNeuromorphicLine};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicI32, Ordering};

/// NeuronGuardTrainerField
/// Manages the pre-allocated, 128-byte aligned neuromorphic memory matrix for training.
pub struct NeuronGuardTrainerField {
    pub sensory_count: usize,
    pub motor_count: usize,
    pub lines: Vec<HighDensityNeuromorphicLine>,
    /// Variable-length overflow store, one growable list per sensory line.
    /// The 24-slot inline core holds each token's strongest successors (cache-resident);
    /// the long tail of additional `(target_id, weight)` associations lives here. This is what
    /// gives each neuron a variable fan-out: "galaxy" may use 5 connections total while "the"
    /// may hold hundreds, without paying for unused slots.
    pub overflow: Vec<Vec<(u16, i16)>>,
    pub potentials: Vec<AtomicI32>,
    pub macro_potentials: Vec<AtomicI32>, // Added for fast hierarchical WTA search
    pub active_indices: Vec<u32>,         // Added to track active potentials in real-time
}

/// Hard ceiling on addressable neurons. `target_ids` is a `u16`, so any successor token must fit
/// in [0, 65535]. Allocating beyond this is pure waste: those neurons can never be the target of
/// a learned connection. We clamp to this bound to prevent multi-gigabyte allocations of dead,
/// unreachable memory from mis-configured vocab sizes.
const MAX_ADDRESSABLE_NEURONS: usize = u16::MAX as usize + 1; // 65,536

impl NeuronGuardTrainerField {
    /// Allocates a flat, contiguous block of memory for sensory and motor neurons.
    ///
    /// `sensory_count` / `motor_count` are clamped to `MAX_ADDRESSABLE_NEURONS` because synaptic
    /// targets are stored as `u16`. Requesting more (e.g. the old 1M/8M "tiers") only produced
    /// empty, unreachable neurons and inflated RSS without adding a single usable connection.
    pub fn new(sensory_count: usize, motor_count: usize) -> Self {
        let requested_sensory = sensory_count;
        let requested_motor = motor_count;
        let sensory_count = sensory_count.min(MAX_ADDRESSABLE_NEURONS);
        let motor_count = motor_count.min(MAX_ADDRESSABLE_NEURONS);
        if requested_sensory > sensory_count || requested_motor > motor_count {
            eprintln!(
                "\u{26a0}\u{fe0f}  NeuronGuard: requested vocab ({} sensory / {} motor) exceeds the \
                 u16 addressable limit of {}. Clamping. Token IDs above {} cannot be stored as \
                 synaptic targets, so the extra neurons would be unreachable dead memory.",
                requested_sensory,
                requested_motor,
                MAX_ADDRESSABLE_NEURONS,
                MAX_ADDRESSABLE_NEURONS - 1
            );
        }

        let mut lines = Vec::with_capacity(sensory_count);
        for _ in 0..sensory_count {
            lines.push(HighDensityNeuromorphicLine::new(15));
        }

        // Overflow lists start empty; they only allocate when a line exceeds its 24-slot core.
        let mut overflow = Vec::with_capacity(sensory_count);
        for _ in 0..sensory_count {
            overflow.push(Vec::new());
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

        let active_indices = Vec::with_capacity(10000);

        Self {
            sensory_count,
            motor_count,
            lines,
            overflow,
            potentials,
            macro_potentials,
            active_indices,
        }
    }

    /// Applies a Hebbian charge to the connection `xt -> target`, transparently spanning the
    /// cache-aligned hot core and the variable-length overflow store. No association is ever
    /// silently dropped: weak newcomers and evicted residents are routed to overflow.
    fn potentiate(&mut self, xt: usize, target_id: u16, charge: i16) {
        match self.lines[xt].adjust_synapse_core(target_id, charge) {
            AdjustResult::Applied => {}
            AdjustResult::Overflow => {
                Self::overflow_add(&mut self.overflow[xt], target_id, charge);
            }
            AdjustResult::Evicted {
                target_id: evicted_target,
                weight: evicted_weight,
            } => {
                // The newcomer is now resident in the core; the displaced (stronger-than-average
                // but now second-tier) resident moves to overflow so its history is preserved.
                Self::overflow_add(&mut self.overflow[xt], evicted_target, evicted_weight);
            }
        }
    }

    /// Adds or accumulates a `(target_id, weight)` pair in a line's overflow store.
    #[inline]
    fn overflow_add(store: &mut Vec<(u16, i16)>, target_id: u16, charge: i16) {
        for entry in store.iter_mut() {
            if entry.0 == target_id {
                entry.1 = entry.1.saturating_add(charge);
                return;
            }
        }
        store.push((target_id, charge));
    }

    /// Resets all potentials to zero.
    pub fn reset_potentials(&mut self) {
        for pot in &self.potentials {
            pot.store(0, Ordering::Relaxed);
        }
        for m_pot in &self.macro_potentials {
            m_pot.store(0, Ordering::Relaxed);
        }
        self.active_indices.clear();
    }

    /// Executes a single-pass Spike-Driven Hebbian Plasticity training step on a stream of token indices.
    /// Completely bypasses the processor's floating-point ALUs.
    ///
    /// Each adjacent `(xt -> xt_next)` pair potentiates the corresponding synapse. Because the
    /// line now owns a variable-length overflow store, every distinct successor a token sees is
    /// retained and its strength reflects how often that transition occurred in the corpus. This
    /// turns the field into a faithful weighted bigram graph rather than a churning 24-slot cache.
    pub fn train_stream_step_sync(&mut self, token_indices: Vec<u32>) {
        if token_indices.len() < 2 {
            return;
        }

        for t in 0..token_indices.len() - 1 {
            let xt = token_indices[t] as usize;
            let xt_next = token_indices[t + 1];

            if xt >= self.sensory_count || (xt_next as usize) >= self.motor_count {
                continue;
            }

            // Hebbian potentiation: reinforce the observed transition. The small per-observation
            // charge accumulates across the corpus, so frequent successors end up with the
            // largest weights and naturally win during weighted sampling. We deliberately do NOT
            // apply any depression here: penalizing non-observed predictions was eroding the very
            // bigram statistics the model depends on.
            self.potentiate(xt, xt_next as u16, 4);
        }
    }

    /// Collects every learned successor `(target_id, weight)` for a token, merging the
    /// cache-aligned hot core with the variable-length overflow store. Weights are clamped to be
    /// strictly positive so they can act directly as sampling masses.
    pub fn successors(&self, xt: usize) -> Vec<(u16, i32)> {
        if xt >= self.lines.len() {
            return Vec::new();
        }
        let line = &self.lines[xt];
        let mut out: Vec<(u16, i32)> = Vec::with_capacity(24 + self.overflow[xt].len());
        for j in 0..24 {
            let w = line.synapses_weights[j];
            if w > 0 {
                out.push((line.target_ids[j], w as i32));
            }
        }
        for &(target_id, w) in &self.overflow[xt] {
            if w > 0 {
                out.push((target_id, w as i32));
            }
        }
        out
    }

    /// Samples the next token directly from a token's learned successor distribution, using
    /// temperature scaling and optional top-k truncation. This replaces the old global-potential
    /// field sampling, which mixed unrelated activations together and produced incoherent output.
    ///
    /// Returns `None` if the token has no learned successors (e.g. an unseen final token), letting
    /// the caller decide how to fall back.
    pub fn sample_next_token(
        &self,
        xt: usize,
        temperature: f32,
        top_k: usize,
        rng_uniform: f32,
    ) -> Option<u32> {
        let mut cands = self.successors(xt);
        if cands.is_empty() {
            return None;
        }

        // Top-k truncation: keep only the strongest successors.
        if top_k > 0 && cands.len() > top_k {
            cands.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            cands.truncate(top_k);
        }

        // Temperature-scaled softmax over the (positive) learned weights.
        let temp = temperature.max(1e-3);
        let max_w = cands.iter().map(|&(_, w)| w).max().unwrap_or(0) as f32;
        let mut probs: Vec<f32> = cands
            .iter()
            .map(|&(_, w)| (((w as f32) - max_w) / (max_w.max(1.0) * temp)).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return Some(cands[0].0 as u32);
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Inverse-CDF sample using the caller-provided uniform draw in [0, 1).
        let mut acc = 0.0;
        let r = rng_uniform.clamp(0.0, 1.0 - f32::EPSILON);
        for (idx, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return Some(cands[idx].0 as u32);
            }
        }
        Some(cands[cands.len() - 1].0 as u32)
    }

    /// Serializes and writes the synaptic matrix to a base64-encoded text file using a
    /// self-describing, variable-length format. Each line is encoded as:
    ///   `[u32 count][ (u16 target, i16 weight) * count ]`
    /// covering the merged core + overflow successors, so per-token fan-out is preserved exactly.
    ///
    /// A short magic header (`NGV2`) plus the line count lets the loader reject incompatible
    /// (legacy fixed-size) model cards with a clear message instead of corrupting state.
    pub fn save_weights_to_b64(&self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(MAGIC);
        raw.extend_from_slice(&(self.lines.len() as u32).to_le_bytes());

        for xt in 0..self.lines.len() {
            let succ = self.successors(xt);
            raw.extend_from_slice(&(succ.len() as u32).to_le_bytes());
            for (target_id, weight) in succ {
                raw.extend_from_slice(&target_id.to_le_bytes());
                // Clamp to i16 range for compact on-disk storage.
                let w = weight.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                raw.extend_from_slice(&w.to_le_bytes());
            }

            // Flush in ~96 KB chunks (aligned to 3 bytes for clean base64) to bound memory.
            if raw.len() >= 96_000 {
                let aligned = raw.len() - (raw.len() % 3);
                let b64 = base64_encode(&raw[..aligned]);
                file.write_all(b64.as_bytes())?;
                raw.drain(..aligned);
            }
        }

        if !raw.is_empty() {
            let b64 = base64_encode(&raw);
            file.write_all(b64.as_bytes())?;
        }

        Ok(())
    }

    /// Loads a variable-length model card produced by `save_weights_to_b64`. The strongest 24
    /// successors per token are reinstated into the cache-aligned hot core; the remainder is
    /// placed in the overflow store, reproducing the in-memory layout deterministically.
    pub fn load_weights_from_b64(&mut self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::open(path)?;
        let mut b64 = String::new();
        file.read_to_string(&mut b64)?;
        let bytes = base64_decode(&b64);

        let invalid =
            |msg: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string());

        if bytes.len() < 8 || &bytes[0..4] != MAGIC {
            return Err(invalid(
                "Unrecognized or legacy model card. Please retrain with the current version (the \
                 synaptic format changed to support variable connections per neuron).",
            ));
        }

        let line_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if line_count != self.lines.len() {
            return Err(invalid(&format!(
                "Synaptic weights file mismatch: file has {} lines, field expects {}",
                line_count,
                self.lines.len()
            )));
        }

        let mut offset = 8usize;
        for xt in 0..self.lines.len() {
            self.lines[xt] = HighDensityNeuromorphicLine::new(15);
            self.overflow[xt].clear();

            if offset + 4 > bytes.len() {
                return Err(invalid("Truncated model card: missing successor count"));
            }
            let count = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            for _ in 0..count {
                if offset + 4 > bytes.len() {
                    return Err(invalid("Truncated model card: missing successor entry"));
                }
                let target_id = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                offset += 2;
                let weight = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                offset += 2;
                // Route through the core/overflow promotion logic so the strongest 24 land in
                // cache-aligned memory just as they did during training.
                self.potentiate(xt, target_id, weight);
            }
        }

        Ok(())
    }

    /// Processes a stream of token indices synchronously for inference. Retained for compatibility;
    /// generation now samples directly from learned successors via `sample_next_token`.
    pub fn process_step_sync(&mut self, token_indices: Vec<u32>) {
        for &xt in &token_indices {
            let xt = xt as usize;
            if xt >= self.sensory_count {
                continue;
            }
            let line = &self.lines[xt];
            for j in 0..24 {
                let weight = line.synapses_weights[j] as i32;
                if weight != 0 {
                    let target_token_id = line.target_ids[j] as usize;
                    let prev =
                        self.potentials[target_token_id].fetch_add(weight, Ordering::Relaxed);
                    if prev == 0 {
                        self.active_indices.push(target_token_id as u32);
                    }
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

    /// Forces an immediate Hebbian integer addition into a targeted cell address
    pub fn potentiate_synapse_sync(&mut self, source: usize, target: usize, weight_delta: i16) {
        if source < self.sensory_count && target < self.motor_count {
            self.potentiate(source, target as u16, weight_delta);
        }
    }

    /// Directly writes to the TPI accumulator register for manual testing
    pub fn inject_potential_sync(&mut self, target_node: usize, voltage: i16) {
        if target_node < self.motor_count {
            let prev = self.potentials[target_node].fetch_add(voltage as i32, Ordering::Relaxed);
            if prev == 0 && voltage != 0 {
                self.active_indices.push(target_node as u32);
            }
        }
    }

    /// Pulls the raw synaptic layout sorted from highest weight down
    pub fn get_row_synapses_sync(&self, row_idx: usize) -> Vec<(usize, i16)> {
        if row_idx < self.sensory_count {
            let line = &self.lines[row_idx];
            let mut synapses = Vec::new();

            for j in 0..24 {
                let w = line.synapses_weights[j];
                if w > 0 {
                    synapses.push((line.target_ids[j] as usize, w));
                }
            }
            for &(target_id, w) in &self.overflow[row_idx] {
                if w > 0 {
                    synapses.push((target_id as usize, w));
                }
            }

            // Sort from highest weight down
            synapses.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            synapses
        } else {
            Vec::new()
        }
    }
}


/// Magic header identifying the variable-length ("NGV2") synaptic model-card format.
const MAGIC: &[u8; 4] = b"NGV2";

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
        let mut trainer = NeuronGuardTrainerField::new(100, 100);
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

        // Synaptic pathway from 2 to 5 should be potentiated with one observation's charge.
        assert_eq!(trainer.lines[2].synapses_weights[0], 4);
        assert_eq!(trainer.lines[2].target_ids[0], 5);

        // Repeated observations accumulate strength.
        trainer.train_stream_step_sync(vec![2, 5]);
        assert_eq!(trainer.lines[2].synapses_weights[0], 8);
    }

    #[test]
    fn test_variable_fan_out_overflow() {
        // A token observed before 30 distinct successors should retain ALL of them:
        // 24 in the cache-aligned core and the remaining 6 in the overflow store.
        let mut trainer = NeuronGuardTrainerField::new(100, 100);
        for next in 1..=30u32 {
            trainer.train_stream_step_sync(vec![0, next]);
        }

        let succ = trainer.successors(0);
        assert_eq!(succ.len(), 30, "all 30 successors must be retained");
        assert!(
            !trainer.overflow[0].is_empty(),
            "overflow store must be used"
        );

        // A rare token with few successors stays tiny (no overflow allocation).
        trainer.train_stream_step_sync(vec![50, 51]);
        assert_eq!(trainer.successors(50).len(), 1);
        assert!(trainer.overflow[50].is_empty());
    }

    #[test]
    fn test_variable_weights_roundtrip() {
        let mut trainer = NeuronGuardTrainerField::new(100, 100);
        for next in 1..=30u32 {
            // Vary frequency so weights differ across successors.
            for _ in 0..next {
                trainer.train_stream_step_sync(vec![0, next]);
            }
        }

        let dir = std::env::temp_dir();
        let path = dir.join("ng_test_weights.txt");
        let path_str = path.to_str().unwrap();
        trainer.save_weights_to_b64(path_str).unwrap();

        let mut loaded = NeuronGuardTrainerField::new(100, 100);
        loaded.load_weights_from_b64(path_str).unwrap();

        let mut a = trainer.successors(0);
        let mut b = loaded.successors(0);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "successor distribution must survive serialization");

        let _ = std::fs::remove_file(path_str);
    }

    #[test]
    fn test_legacy_model_card_rejected() {
        let dir = std::env::temp_dir();
        let path = dir.join("ng_legacy_weights.txt");
        let path_str = path.to_str().unwrap();
        // Write a blob without the NGV2 magic header.
        std::fs::write(path_str, base64_encode(b"OLDFORMATDATA....")).unwrap();

        let mut loaded = NeuronGuardTrainerField::new(10, 10);
        assert!(loaded.load_weights_from_b64(path_str).is_err());
        let _ = std::fs::remove_file(path_str);
    }

    #[test]
    fn test_base64_encode() {
        let data = b"hello";
        let encoded = base64_encode(data);
        assert_eq!(encoded, "aGVsbG8=");
    }
}
