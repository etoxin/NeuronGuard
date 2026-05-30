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

/// High-Resolution Cache-Aligned Neuromorphic Line Subsystem.
/// Enforces a strict 128-byte footprint to maximize CPU L1/L2 prefetch hit ratios.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct MaxRangeNeuromorphicLine {
    /// 32 high-precision synapses tracking target token pathways with deep statistical headroom.
    /// Consumes exactly 64 bytes (32 elements * 2 bytes each). Zero unpacking overhead.
    pub synapses_weights: [i16; 32],

    /// Relative offset pointer for Central Pattern Generator backward routing (4 Bytes)
    pub loopback_address: u32,

    /// Current accumulated potential headroom (4 Bytes)
    pub local_potential: i32,

    /// Dynamic activation threshold before a spike event is triggered (4 Bytes)
    pub activation_threshold: i32,

    /// Remaining lingering energy amplitude inside the recurrent loop (1 Byte)
    pub loopback_energy: u8,

    /// Explicit padding array ensuring the total struct size hits exactly 128 bytes on silicon.
    /// 128 - (64 + 4 + 4 + 4 + 1) = 51 bytes of trailing block safety.
    /// By ordering fields by alignment, we eliminate compiler-inserted padding.
    pub _padding: [u8; 51],
}

impl MaxRangeNeuromorphicLine {
    /// Instantiate a completely sterile, cache-aligned neural line
    pub fn new(initial_threshold: i32) -> Self {
        Self {
            synapses_weights: [0; 32],
            loopback_address: 0,
            loopback_energy: 0,
            local_potential: 0,
            activation_threshold: initial_threshold,
            _padding: [0; 51],
        }
    }

    /// Single-pass Hebbian potentiation step using fast hardware-level saturating addition
    #[inline(always)]
    pub fn potentiate_synapse(&mut self, index: usize, adjustment: i16) {
        if index < 32 {
            // saturating_add guarantees the value locks at +32767 instead of crashing via overflow
            self.synapses_weights[index] = self.synapses_weights[index].saturating_add(adjustment);
        }
    }

    /// Single-pass Hebbian depression step using fast hardware-level saturating subtraction
    #[inline(always)]
    pub fn depress_synapse(&mut self, index: usize, adjustment: i16) {
        if index < 32 {
            // saturating_sub guarantees the value locks at -32768 instead of rolling over
            self.synapses_weights[index] = self.synapses_weights[index].saturating_sub(adjustment);
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
        assert_eq!(size_of::<MaxRangeNeuromorphicLine>(), 128);
        assert_eq!(align_of::<MaxRangeNeuromorphicLine>(), 128);
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
