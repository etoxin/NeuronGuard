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

use parking_lot::RwLock;
use std::alloc::{alloc_zeroed, dealloc, Layout};

// Configuration Bounds
pub const MAX_THREADS: usize = 8; // Locked directly to target CPU core architecture

/// ThreadBoundedNeuron
/// Spatially aligned to exactly 64 bytes to fill a standard CPU cache line.
/// This provides fixed-width addressing and keeps independent neuron payloads
/// from sharing the same cache line.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct ThreadBoundedNeuron {
    pub token_id: u32,                         // 4 Bytes: Unique symbol key
    pub active_connections: u32,               // 4 Bytes: Actual count (<= MAX_THREADS)
    pub target_neuron_ids: [u32; MAX_THREADS], // 32 Bytes: Downstream destination buckets
    pub weight_modifiers: [i16; MAX_THREADS],  // 16 Bytes: Localized connection strengths
    pub padding: [u8; 8],                      // 8 Bytes: Structure alignment round-out
}

impl ThreadBoundedNeuron {
    /// Updates an existing connection or adds a new one.
    /// If capacity is reached, executes autonomous least-significant eviction (hardware self-pruning).
    pub fn update_or_add_connection(&mut self, target_id: u32, weight_delta: i16) {
        // First, check if the connection already exists
        for i in 0..self.active_connections as usize {
            if self.target_neuron_ids[i] == target_id {
                self.weight_modifiers[i] = self.weight_modifiers[i].saturating_add(weight_delta);
                return;
            }
        }

        // If it doesn't exist, check if we have space
        if (self.active_connections as usize) < MAX_THREADS {
            let idx = self.active_connections as usize;
            self.target_neuron_ids[idx] = target_id;
            self.weight_modifiers[idx] = weight_delta;
            self.active_connections += 1;
        } else {
            // No space! Execute autonomous least-significant eviction
            // Find the weakest connection (value closest to zero)
            let mut weakest_idx = 0;
            let mut weakest_val = self.weight_modifiers[0].unsigned_abs();

            for i in 1..MAX_THREADS {
                let val = self.weight_modifiers[i].unsigned_abs();
                if val < weakest_val {
                    weakest_val = val;
                    weakest_idx = i;
                }
            }

            // Evict weakest connection
            self.target_neuron_ids[weakest_idx] = target_id;
            self.weight_modifiers[weakest_idx] = weight_delta;
        }
    }
}

/// ThreadBoundedNeuronField
/// Manages a flat, contiguous block of memory for ThreadBoundedNeurons.
pub struct ThreadBoundedNeuronField {
    storage: *mut ThreadBoundedNeuron,
    size: usize,
    mmap: Option<memmap2::MmapMut>,
    locks: Box<[RwLock<()>]>,
}

impl ThreadBoundedNeuronField {
    /// Allocates a flat, contiguous block of memory for `size` neurons, zero-initialized.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "ThreadBoundedNeuronField size must be positive");
        let layout = Layout::array::<ThreadBoundedNeuron>(size)
            .expect("Failed to create memory layout for ThreadBoundedNeuronField");

        let storage = unsafe {
            let ptr = alloc_zeroed(layout) as *mut ThreadBoundedNeuron;
            if ptr.is_null() {
                panic!("Failed to allocate memory for ThreadBoundedNeuronField");
            }
            ptr
        };

        let field = Self {
            storage,
            size,
            mmap: None,
            locks: Self::make_locks(size),
        };
        for id in 0..size {
            field.with_neuron_mut(id, |neuron| neuron.token_id = id as u32);
        }
        field
    }

    /// Creates a field directly from a memory-mapped file for zero-copy loading.
    pub fn from_mmap(mut mmap: memmap2::MmapMut, size: usize) -> Self {
        assert!(size > 0, "ThreadBoundedNeuronField size must be positive");
        assert_eq!(
            mmap.len(),
            Self::byte_len_for(size),
            "Memory-mapped model length does not match neuron count"
        );
        let storage = mmap.as_mut_ptr() as *mut ThreadBoundedNeuron;
        Self {
            storage,
            size,
            mmap: Some(mmap),
            locks: Self::make_locks(size),
        }
    }

    fn make_locks(size: usize) -> Box<[RwLock<()>]> {
        (0..size)
            .map(|_| RwLock::new(()))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn byte_len_for(size: usize) -> usize {
        size.checked_mul(std::mem::size_of::<ThreadBoundedNeuron>())
            .expect("Neuron field byte length overflow")
    }

    /// Reads a neuron while holding its shared lock.
    #[cfg_attr(not(feature = "extension-module"), allow(dead_code))]
    pub(crate) fn with_neuron<R>(
        &self,
        id: usize,
        read: impl FnOnce(&ThreadBoundedNeuron) -> R,
    ) -> Option<R> {
        if id >= self.size {
            return None;
        }
        let _guard = self.locks[id].read();
        // SAFETY: `id` is in bounds, allocation and mmap storage are both aligned for
        // `ThreadBoundedNeuron`, and the read lock excludes every mutable accessor.
        let neuron = unsafe { &*self.storage.add(id) };
        Some(read(neuron))
    }

    /// Mutates a neuron while holding its exclusive lock.
    pub(crate) fn with_neuron_mut<R>(
        &self,
        id: usize,
        update: impl FnOnce(&mut ThreadBoundedNeuron) -> R,
    ) -> Option<R> {
        if id >= self.size {
            return None;
        }
        let _guard = self.locks[id].write();
        // SAFETY: `id` is in bounds, allocation and mmap storage are both aligned for
        // `ThreadBoundedNeuron`, and the write lock excludes all other accessors.
        let neuron = unsafe { &mut *self.storage.add(id) };
        Some(update(neuron))
    }

    /// Copies a consistent snapshot of the flat neuron payload.
    #[cfg(feature = "extension-module")]
    pub(crate) fn snapshot_bytes(&self) -> Vec<u8> {
        let _guards: Vec<_> = self.locks.iter().map(RwLock::read).collect();
        // SAFETY: all neuron read locks are held, so the allocation cannot be mutated
        // while the byte slice is copied. The allocation spans exactly `byte_len` bytes.
        unsafe {
            std::slice::from_raw_parts(self.storage.cast::<u8>(), Self::byte_len_for(self.size))
                .to_vec()
        }
    }
}

impl Drop for ThreadBoundedNeuronField {
    fn drop(&mut self) {
        if self.mmap.is_none() {
            let layout = Layout::array::<ThreadBoundedNeuron>(self.size)
                .expect("Failed to create layout for deallocation");
            unsafe {
                dealloc(self.storage as *mut u8, layout);
            }
        }
    }
}

// SAFETY: the raw pointer is private and every dereference is protected by the
// corresponding per-neuron `RwLock`. Ownership prevents `Drop` racing access.
unsafe impl Send for ThreadBoundedNeuronField {}
// SAFETY: shared access can only reach storage through the locked closure APIs.
unsafe impl Sync for ThreadBoundedNeuronField {}

// Instant, allocation-free feature compression
pub fn tokenize_features(metric_a: f64, metric_b: f64, metric_c: f64) -> u64 {
    let mut token: u64 = 0;

    let bucket_a = if metric_a > 20.0 {
        7
    } else if metric_a > 15.0 {
        4
    } else {
        1
    };
    token |= bucket_a;

    let bucket_b = if metric_b > 1200.0 {
        7
    } else if metric_b > 800.0 {
        4
    } else {
        1
    };
    token |= bucket_b << 8;

    let bucket_c = if metric_c > 0.25 {
        7
    } else if metric_c > 0.10 {
        4
    } else {
        1
    };
    token |= bucket_c << 16;

    token
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_neuron_size_and_alignment() {
        assert_eq!(size_of::<ThreadBoundedNeuron>(), 64);
        assert_eq!(align_of::<ThreadBoundedNeuron>(), 64);
    }

    #[test]
    fn test_tokenize_features() {
        let token = tokenize_features(25.0, 1500.0, 0.30);
        assert_eq!(token, 460551);
    }

    #[test]
    fn test_field_allocation_and_access() {
        let field = ThreadBoundedNeuronField::new(10);
        field.with_neuron_mut(3, |n| {
            n.token_id = 123;
            n.active_connections = 2;
            n.target_neuron_ids[0] = 5;
            n.weight_modifiers[0] = 10;
        });
        field.with_neuron(3, |n_check| {
            assert_eq!(n_check.token_id, 123);
            assert_eq!(n_check.active_connections, 2);
            assert_eq!(n_check.target_neuron_ids[0], 5);
            assert_eq!(n_check.weight_modifiers[0], 10);
        });
    }

    #[test]
    fn test_autonomous_eviction() {
        let mut neuron = ThreadBoundedNeuron {
            token_id: 1,
            active_connections: 8,
            target_neuron_ids: [0, 1, 2, 3, 4, 5, 6, 7],
            weight_modifiers: [10, 20, 30, 2, 50, 60, 70, 80],
            padding: [0; 8],
        };

        neuron.update_or_add_connection(99, 15);

        assert_eq!(neuron.target_neuron_ids[3], 99);
        assert_eq!(neuron.weight_modifiers[3], 15);
        assert_eq!(neuron.active_connections, 8);
    }

    #[test]
    fn test_autonomous_eviction_with_min_weight() {
        let mut neuron = ThreadBoundedNeuron {
            token_id: 1,
            active_connections: 8,
            target_neuron_ids: [0, 1, 2, 3, 4, 5, 6, 7],
            weight_modifiers: [10, 20, 30, i16::MIN, 50, 60, 70, 80],
            padding: [0; 8],
        };

        // Add a new connection. It should NOT panic on i16::MIN.
        // It should evict index 0 (weight 10 is closest to 0, since i16::MIN has absolute value 32768).
        neuron.update_or_add_connection(99, 15);

        assert_eq!(neuron.target_neuron_ids[0], 99);
        assert_eq!(neuron.weight_modifiers[0], 15);
    }

    #[test]
    fn test_saturating_weight_updates() {
        let mut neuron = ThreadBoundedNeuron {
            token_id: 1,
            active_connections: 1,
            target_neuron_ids: [42, 0, 0, 0, 0, 0, 0, 0],
            weight_modifiers: [32760, 0, 0, 0, 0, 0, 0, 0],
            padding: [0; 8],
        };

        // Adding 10 to 32760 should saturate at i16::MAX (32767)
        neuron.update_or_add_connection(42, 10);
        assert_eq!(neuron.weight_modifiers[0], i16::MAX);

        // Subtracting 10 from -32760 should saturate at i16::MIN (-32768)
        neuron.weight_modifiers[0] = -32760;
        neuron.update_or_add_connection(42, -10);
        assert_eq!(neuron.weight_modifiers[0], i16::MIN);
    }

    #[test]
    fn test_locked_access_out_of_bounds() {
        let field = ThreadBoundedNeuronField::new(5);
        assert!(field.with_neuron(5, |_| ()).is_none());
        assert!(field.with_neuron_mut(100, |_| ()).is_none());
    }

    #[test]
    fn test_concurrent_updates_are_lossless() {
        use std::sync::Arc;
        use std::thread;

        let field = Arc::new(ThreadBoundedNeuronField::new(1));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let field_clone = Arc::clone(&field);
            handles.push(thread::spawn(move || {
                for _ in 0..1_000 {
                    field_clone.with_neuron_mut(0, |neuron| neuron.update_or_add_connection(0, 1));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        field.with_neuron(0, |neuron| {
            assert_eq!(neuron.active_connections, 1);
            assert_eq!(neuron.target_neuron_ids[0], 0);
            assert_eq!(neuron.weight_modifiers[0], 4_000);
        });
    }

    #[test]
    fn test_tokenize_features_boundaries() {
        // Test low values
        assert_eq!(
            tokenize_features(10.0, 500.0, 0.05),
            1 | (1 << 8) | (1 << 16)
        );

        // Test medium values
        assert_eq!(
            tokenize_features(18.0, 1000.0, 0.15),
            4 | (4 << 8) | (4 << 16)
        );

        // Test high values
        assert_eq!(
            tokenize_features(25.0, 1500.0, 0.30),
            7 | (7 << 8) | (7 << 16)
        );
    }
}
