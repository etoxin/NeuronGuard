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

use crossbeam_channel::{unbounded, Sender};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

// Configuration Bounds
pub const MAX_THREADS: usize = 8; // Locked directly to target CPU core architecture

/// ThreadBoundedNeuron
/// Spatially aligned to exactly 64 bytes to fill a standard CPU cache line.
/// Eliminates false sharing and guarantees deterministic hardware pre-fetching.
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
    pub storage: *mut ThreadBoundedNeuron,
    pub size: usize,
}

impl ThreadBoundedNeuronField {
    /// Allocates a flat, contiguous block of memory for `size` neurons, zero-initialized.
    pub fn new(size: usize) -> Self {
        let layout = Layout::array::<ThreadBoundedNeuron>(size)
            .expect("Failed to create memory layout for ThreadBoundedNeuronField");

        let storage = unsafe {
            let ptr = alloc_zeroed(layout) as *mut ThreadBoundedNeuron;
            if ptr.is_null() {
                panic!("Failed to allocate memory for ThreadBoundedNeuronField");
            }
            ptr
        };

        Self { storage, size }
    }

    /// Pure pointerless offset arithmetic mapping to Base + ID * 64
    ///
    /// # Safety
    /// This is unsafe because it performs raw pointer arithmetic.
    /// The caller must ensure that the `id` is within bounds.
    pub unsafe fn get_neuron(&self, id: usize) -> &mut ThreadBoundedNeuron {
        if id >= self.size {
            panic!("Neuron ID out of bounds: {} >= {}", id, self.size);
        }
        // self.storage.add(id) calculates: storage_address + (id * size_of::<ThreadBoundedNeuron>())
        // Since ThreadBoundedNeuron is 64 bytes, this is exactly: Base + ID * 64 (or ID << 6)
        &mut *self.storage.add(id)
    }

    /// Acquires a transactional, lock-free lease on the specific 64-byte memory address of the active token.
    /// Uses the first 4 bytes of padding as an AtomicU32 lease flag.
    pub fn try_acquire_lease(&self, id: usize) -> Option<NeuronLease<'_>> {
        if id >= self.size {
            return None;
        }
        unsafe {
            let neuron = self.get_neuron(id);
            let lease_ptr = neuron.padding.as_ptr() as *const AtomicU32;
            let lease_ref = &*lease_ptr;
            if lease_ref
                .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                Some(NeuronLease {
                    neuron_id: id,
                    field: self,
                })
            } else {
                None
            }
        }
    }
}

impl Drop for ThreadBoundedNeuronField {
    fn drop(&mut self) {
        let layout = Layout::array::<ThreadBoundedNeuron>(self.size)
            .expect("Failed to create layout for deallocation");
        unsafe {
            dealloc(self.storage as *mut u8, layout);
        }
    }
}

unsafe impl Send for ThreadBoundedNeuronField {}
unsafe impl Sync for ThreadBoundedNeuronField {}

/// NeuronLease
/// Represents a transactional lease on a specific neuron.
pub struct NeuronLease<'a> {
    pub neuron_id: usize,
    pub field: &'a ThreadBoundedNeuronField,
}

impl<'a> NeuronLease<'a> {
    pub fn neuron(&self) -> &mut ThreadBoundedNeuron {
        unsafe { self.field.get_neuron(self.neuron_id) }
    }
}

impl<'a> Drop for NeuronLease<'a> {
    fn drop(&mut self) {
        unsafe {
            let neuron = self.field.get_neuron(self.neuron_id);
            let lease_ptr = neuron.padding.as_ptr() as *const AtomicU32;
            let lease_ref = &*lease_ptr;
            lease_ref.store(0, Ordering::SeqCst);
        }
    }
}

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
    token |= bucket_a << 0;

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

pub fn evaluate_parallel(
    neuron: &ThreadBoundedNeuron,
    thread_id: usize,
    accumulators: &[AtomicI32],
) {
    if thread_id < neuron.active_connections as usize {
        let target = neuron.target_neuron_ids[thread_id] as usize;
        let weight = neuron.weight_modifiers[thread_id] as i32;

        // Concurrent atomic addition straight into target accumulation state
        if target < accumulators.len() {
            accumulators[target].fetch_add(weight, Ordering::Relaxed);
        }
    }
}

/// ParallelRouter
/// Manages a thread pool that processes neuron evaluations in parallel.
pub struct ParallelRouter {
    senders: Vec<Sender<ThreadBoundedNeuron>>,
    handles: Vec<JoinHandle<()>>,
}

impl ParallelRouter {
    pub fn new(accumulators: Arc<Vec<AtomicI32>>) -> Self {
        let mut senders = Vec::new();
        let mut handles = Vec::new();

        for thread_id in 0..MAX_THREADS {
            let (sender, receiver) = unbounded::<ThreadBoundedNeuron>();
            senders.push(sender);

            let accumulators_clone = Arc::clone(&accumulators);
            let handle = thread::spawn(move || {
                while let Ok(neuron) = receiver.recv() {
                    evaluate_parallel(&neuron, thread_id, &accumulators_clone);
                }
            });
            handles.push(handle);
        }

        Self { senders, handles }
    }

    pub fn broadcast(&self, neuron: ThreadBoundedNeuron) {
        for sender in &self.senders {
            let _ = sender.send(neuron);
        }
    }
}

impl Drop for ParallelRouter {
    fn drop(&mut self) {
        self.senders.clear();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

unsafe impl Send for ParallelRouter {}
unsafe impl Sync for ParallelRouter {}

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
    fn test_field_allocation_and_leap() {
        let field = ThreadBoundedNeuronField::new(10);
        unsafe {
            let n = field.get_neuron(3);
            n.token_id = 123;
            n.active_connections = 2;
            n.target_neuron_ids[0] = 5;
            n.weight_modifiers[0] = 10;

            let n_check = field.get_neuron(3);
            assert_eq!(n_check.token_id, 123);
            assert_eq!(n_check.active_connections, 2);
            assert_eq!(n_check.target_neuron_ids[0], 5);
            assert_eq!(n_check.weight_modifiers[0], 10);
        }
    }

    #[test]
    fn test_neuron_lease() {
        let field = ThreadBoundedNeuronField::new(5);

        let lease1 = field.try_acquire_lease(2);
        assert!(lease1.is_some());

        let lease2 = field.try_acquire_lease(2);
        assert!(lease2.is_none());

        drop(lease1);

        let lease3 = field.try_acquire_lease(2);
        assert!(lease3.is_some());
    }

    #[test]
    fn test_parallel_evaluation() {
        let accumulators = Arc::new((0..10).map(|_| AtomicI32::new(0)).collect::<Vec<_>>());
        let router = ParallelRouter::new(Arc::clone(&accumulators));

        let mut neuron = ThreadBoundedNeuron {
            token_id: 42,
            active_connections: 3,
            target_neuron_ids: [0; MAX_THREADS],
            weight_modifiers: [0; MAX_THREADS],
            padding: [0; 8],
        };
        neuron.target_neuron_ids[0] = 1;
        neuron.weight_modifiers[0] = 5;
        neuron.target_neuron_ids[1] = 2;
        neuron.weight_modifiers[1] = -3;
        neuron.target_neuron_ids[2] = 3;
        neuron.weight_modifiers[2] = 10;

        router.broadcast(neuron);

        std::thread::sleep(std::time::Duration::from_millis(50));

        assert_eq!(accumulators[1].load(Ordering::Relaxed), 5);
        assert_eq!(accumulators[2].load(Ordering::Relaxed), -3);
        assert_eq!(accumulators[3].load(Ordering::Relaxed), 10);
        assert_eq!(accumulators[0].load(Ordering::Relaxed), 0);
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
    fn test_try_acquire_lease_out_of_bounds() {
        let field = ThreadBoundedNeuronField::new(5);
        assert!(field.try_acquire_lease(5).is_none());
        assert!(field.try_acquire_lease(100).is_none());
    }

    #[test]
    #[should_panic(expected = "Neuron ID out of bounds")]
    fn test_get_neuron_out_of_bounds_panic() {
        let field = ThreadBoundedNeuronField::new(5);
        unsafe {
            field.get_neuron(5);
        }
    }

    #[test]
    fn test_concurrent_lease_acquisition() {
        use std::sync::Barrier;

        let field = Arc::new(ThreadBoundedNeuronField::new(1));
        let barrier1 = Arc::new(Barrier::new(4));
        let barrier2 = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let field_clone = Arc::clone(&field);
            let barrier1_clone = Arc::clone(&barrier1);
            let barrier2_clone = Arc::clone(&barrier2);
            handles.push(thread::spawn(move || {
                barrier1_clone.wait();
                let lease = field_clone.try_acquire_lease(0);
                let acquired = lease.is_some();
                barrier2_clone.wait();
                acquired
            }));
        }

        let mut success_count = 0;
        for handle in handles {
            if handle.join().unwrap() {
                success_count += 1;
            }
        }

        // Exactly one thread must have successfully acquired the lease
        assert_eq!(success_count, 1);
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
