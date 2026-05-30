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

use std::sync::atomic::{AtomicI16, Ordering};

/// Enforce strict 64-byte alignment to match standard CPU cache lines.
/// This maximizes L1/L2 data locality and prevents cache line thrashing.
/// This version is aligned to 64 bytes and padded to exactly 128 bytes (two cache lines)
/// to accommodate 256-bit positive and negative synaptic pathways.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct PermanentNeuromorphicLine {
    /// 256 bits representing active excitatory synaptic pathways
    pub synapses_positive: [u32; 8],

    /// 256 bits representing active inhibitory synaptic pathways
    pub synapses_negative: [u32; 8],

    /// Relative offset pointer for Central Pattern Generator backward routing
    pub loopback_address: u32,

    /// Remaining lingering energy amplitude inside the recurrent loop
    pub loopback_energy: u8,

    /// Current integer accumulation potential
    pub local_potential: i16,

    /// Dynamic activation threshold before a spike event is triggered
    pub activation_threshold: i16,

    /// Strict padding to guarantee that instances align perfectly to hardware bounds (128 bytes total).
    /// Taking into account compiler alignment padding (1 byte between loopback_energy and local_potential),
    /// we use exactly 54 bytes of padding.
    pub _padding: [u8; 54],
}

pub struct AtomicPotentialState {
    pub potential: AtomicI16,
}

impl AtomicPotentialState {
    pub fn new(initial: i16) -> Self {
        Self {
            potential: AtomicI16::new(initial),
        }
    }

    pub fn try_lease_and_accumulate(&self, increment: i16) {
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
        assert_eq!(size_of::<PermanentNeuromorphicLine>(), 128);
        assert_eq!(align_of::<PermanentNeuromorphicLine>(), 64);
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
