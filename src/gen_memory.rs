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
    /// Expanded from 32 to 56 slots. Tracks 56 target token pathways natively.
    /// Consumes exactly 112 bytes (56 elements * 2 bytes each). Zero unpacking overhead.
    pub synapses_weights: [i16; 56],

    /// Relative offset pointer for Central Pattern Generator routing (4 Bytes)
    pub loopback_address: u32,

    /// Current accumulated potential (4 Bytes)
    pub local_potential: i32,

    /// Dynamic activation threshold before a spike event triggers (4 Bytes)
    pub activation_threshold: i32,

    /// Remaining lingering energy amplitude inside the recurrent loop (1 Byte)
    pub loopback_energy: u8,

    /// Tightened padding array to hit the 128-byte silicon boundary perfectly.
    /// 128 - (112 + 4 + 4 + 4 + 1) = Exactly 3 bytes of structural margin safety.
    /// By ordering fields by alignment, we eliminate compiler-inserted padding.
    pub _padding: [u8; 3],
}

impl HighDensityNeuromorphicLine {
    /// Instantiate a completely sterile, cache-aligned neural line
    pub fn new(initial_threshold: i32) -> Self {
        Self {
            synapses_weights: [0; 56], // Initialize expanded array
            loopback_address: 0,
            loopback_energy: 0,
            local_potential: 0,
            activation_threshold: initial_threshold,
            _padding: [0; 3],
        }
    }

    /// Single-pass Hebbian synaptic adjustment step using fast hardware-level saturating addition
    #[inline(always)]
    pub fn adjust_synapse(&mut self, index: usize, charge: i16) {
        if index < 56 {
            // Check against new boundary ceiling
            // saturating_add guarantees the value locks at boundaries instead of crashing via overflow
            self.synapses_weights[index] = self.synapses_weights[index].saturating_add(charge);
        }
    }
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
