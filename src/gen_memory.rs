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

use std::sync::atomic::{AtomicI32, Ordering};

/// High-Density Cache-Aligned Neuromorphic Line Subsystem.
/// Enforces a strict 128-byte footprint to maximize CPU L1/L2 prefetch hit ratios.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct HighDensityNeuromorphicLine {
    /// 24 high-precision synapses tracking target token pathways with deep statistical headroom.
    /// Consumes exactly 48 bytes (24 elements * 2 bytes each).
    pub synapses_weights: [i16; 24],

    /// 24 high-precision target token IDs allowing connections to any arbitrary word in the 50,000 vocabulary.
    /// Consumes exactly 48 bytes (24 elements * 2 bytes each).
    pub target_ids: [u16; 24],

    /// Relative offset pointer for Central Pattern Generator routing (4 Bytes)
    pub loopback_address: u32,

    /// Current accumulated potential (4 Bytes)
    pub local_potential: i32,

    /// Dynamic activation threshold before a spike event triggers (4 Bytes)
    pub activation_threshold: i32,

    /// Remaining lingering energy amplitude inside the recurrent loop (1 Byte)
    pub loopback_energy: u8,

    /// Tightened padding array to hit the 128-byte silicon boundary perfectly.
    /// 128 - (48 + 48 + 4 + 4 + 4 + 1) = Exactly 19 bytes of structural margin safety.
    /// By ordering fields by alignment, we eliminate compiler-inserted padding.
    pub _padding: [u8; 19],
}

impl HighDensityNeuromorphicLine {
    /// Instantiate a completely sterile, cache-aligned neural line
    pub fn new(initial_threshold: i32) -> Self {
        Self {
            synapses_weights: [0; 24],
            target_ids: [0; 24],
            loopback_address: 0,
            loopback_energy: 0,
            local_potential: 0,
            activation_threshold: initial_threshold,
            _padding: [0; 19],
        }
    }

    /// Single-pass Hebbian synaptic adjustment step using fast hardware-level saturating addition.
    ///
    /// This operates ONLY on the 24-slot cache-aligned hot core. It returns:
    ///   - `AdjustResult::Applied`  if the target already lived in the core or fit in a free slot.
    ///   - `AdjustResult::Overflow` if the core is full and the new target does not belong here
    ///     (it should be routed to the line's variable-length overflow store instead).
    ///   - `AdjustResult::Evicted { target_id, weight }` if the new target was strong enough to
    ///     displace the weakest resident, in which case the evicted pair should be pushed to the
    ///     overflow store so no learned association is ever lost.
    ///
    /// Unlike the previous design, the hot core is now a *cache* of the strongest synapses rather
    /// than a hard 24-connection ceiling. Variable fan-out lives in the line's overflow store, so
    /// rare words like "galaxy" stay tiny while frequent words like "the" can hold hundreds of
    /// successors.
    #[inline(always)]
    pub fn adjust_synapse_core(&mut self, target_id: u16, charge: i16) -> AdjustResult {
        // 1. Check if the connection already exists in the hot core.
        for i in 0..24 {
            if self.target_ids[i] == target_id && self.synapses_weights[i] != 0 {
                self.synapses_weights[i] = self.synapses_weights[i].saturating_add(charge);
                return AdjustResult::Applied;
            }
        }

        // 2. If it doesn't exist, find an empty slot (weight is 0).
        for i in 0..24 {
            if self.synapses_weights[i] == 0 {
                self.target_ids[i] = target_id;
                self.synapses_weights[i] = charge;
                return AdjustResult::Applied;
            }
        }

        // 3. Core is full. Find the weakest resident.
        let mut weakest_idx = 0;
        let mut weakest_val = self.synapses_weights[0].unsigned_abs();
        for i in 1..24 {
            let val = self.synapses_weights[i].unsigned_abs();
            if val < weakest_val {
                weakest_val = val;
                weakest_idx = i;
            }
        }

        // Only promote the newcomer into the core if it is genuinely stronger than the weakest
        // resident; otherwise it belongs in the overflow store. This keeps the hottest synapses
        // resident in cache-aligned memory while still preserving the long tail.
        if charge.unsigned_abs() > weakest_val {
            let evicted_target = self.target_ids[weakest_idx];
            let evicted_weight = self.synapses_weights[weakest_idx];
            self.target_ids[weakest_idx] = target_id;
            self.synapses_weights[weakest_idx] = charge;
            AdjustResult::Evicted {
                target_id: evicted_target,
                weight: evicted_weight,
            }
        } else {
            AdjustResult::Overflow
        }
    }

    /// Backwards-compatible helper retained for tests and the legacy single-line API.
    /// Performs the same core adjustment but silently discards overflow routing information.
    #[inline(always)]
    pub fn adjust_synapse(&mut self, target_id: u16, charge: i16) {
        let _ = self.adjust_synapse_core(target_id, charge);
    }
}

/// Outcome of a hot-core synaptic adjustment, used to drive variable-length overflow routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjustResult {
    /// The adjustment was fully absorbed by the 24-slot cache-aligned core.
    Applied,
    /// The core is full and the newcomer was weaker than every resident; route it to overflow.
    Overflow,
    /// The newcomer displaced a weaker resident; push the returned pair to overflow.
    Evicted { target_id: u16, weight: i16 },
}

pub struct AtomicPotentialState {
    pub potential: AtomicI32,
}

impl AtomicPotentialState {
    pub fn new(initial: i32) -> Self {
        Self {
            potential: AtomicI32::new(initial),
        }
    }

    pub fn try_lease_and_accumulate(&self, increment: i32) {
        let mut current = self.potential.load(Ordering::Relaxed);
        loop {
            let target = current.saturating_add(increment);
            match self.potential.compare_exchange_weak(
                current,
                target,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_permanent_line_size_and_alignment() {
        assert_eq!(size_of::<HighDensityNeuromorphicLine>(), 128);
        assert_eq!(align_of::<HighDensityNeuromorphicLine>(), 128);
    }

    #[test]
    fn test_atomic_potential_state_cas() {
        let state = AtomicPotentialState::new(10);
        state.try_lease_and_accumulate(5);
        assert_eq!(state.potential.load(Ordering::Relaxed), 15);

        state.try_lease_and_accumulate(-20);
        assert_eq!(state.potential.load(Ordering::Relaxed), -5);
    }
}
