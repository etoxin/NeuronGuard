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

use std::alloc::{alloc_zeroed, dealloc, Layout};

// Force alignment to 16 bytes in memory
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GuardedNeuron {
    pub potential: f32, // 4 bytes
    pub threshold: f32, // 4 bytes
    pub target_id: u32, // 4 bytes
    pub weight: f32,    // 4 bytes
} // Total = 16 bytes

pub struct NeuronField {
    pub storage: *mut GuardedNeuron,
    pub size: usize,
}

impl NeuronField {
    /// Allocates a flat, contiguous block of memory for `size` neurons, zero-initialized.
    pub fn new(size: usize) -> Self {
        // Create a memory layout for an array of GuardedNeurons
        let layout = Layout::array::<GuardedNeuron>(size)
            .expect("Failed to create memory layout for NeuronField");

        // Allocate zeroed memory so all floats are 0.0 and integers are 0
        let storage = unsafe {
            let ptr = alloc_zeroed(layout) as *mut GuardedNeuron;
            if ptr.is_null() {
                panic!("Failed to allocate memory for NeuronField");
            }
            ptr
        };

        Self { storage, size }
    }

    /// Pure pointerless offset arithmetic mapping to Base + ID * 16
    ///
    /// # Safety
    /// This is unsafe because it performs raw pointer arithmetic.
    /// The caller must ensure that the `id` is within bounds.
    pub unsafe fn get_neuron(&self, id: usize) -> &mut GuardedNeuron {
        if id >= self.size {
            panic!("Neuron ID out of bounds: {} >= {}", id, self.size);
        }
        // self.storage.add(id) calculates: storage_address + (id * size_of::<GuardedNeuron>())
        // Since GuardedNeuron is 16 bytes, this is exactly: Base + ID * 16
        &mut *self.storage.add(id)
    }
}

// Implement Drop to prevent memory leaks (like a destructor in C++ or manually freeing memory)
impl Drop for NeuronField {
    fn drop(&mut self) {
        let layout = Layout::array::<GuardedNeuron>(self.size)
            .expect("Failed to create layout for deallocation");
        unsafe {
            dealloc(self.storage as *mut u8, layout);
        }
    }
}

// --- Thread Safety Declarations ---
// In Rust, raw pointers (*mut T) are NOT thread-safe by default because the compiler
// doesn't know how you intend to use them.
// Since we will have multiple threads reading and writing to different neurons in the field
// without overlapping, we must explicitly tell the compiler that it is safe to send
// and share this struct across threads.
unsafe impl Send for NeuronField {}
unsafe impl Sync for NeuronField {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_neuron_size_is_exactly_16_bytes() {
        // This validates Phase 1: Compile-Time Size Enforcement
        assert_eq!(size_of::<GuardedNeuron>(), 16);
    }

    #[test]
    fn test_neuron_field_allocation_and_access() {
        let field = NeuronField::new(10);
        assert_eq!(field.size, 10);

        unsafe {
            let n0 = field.get_neuron(0);
            n0.potential = 1.23;
            n0.threshold = 4.56;
            n0.target_id = 99;
            n0.weight = 0.88;

            let n0_check = field.get_neuron(0);
            assert_eq!(n0_check.potential, 1.23);
            assert_eq!(n0_check.threshold, 4.56);
            assert_eq!(n0_check.target_id, 99);
            assert_eq!(n0_check.weight, 0.88);
        }
    }

    #[test]
    #[should_panic(expected = "Neuron ID out of bounds")]
    fn test_neuron_field_bounds_check() {
        let field = NeuronField::new(5);
        unsafe {
            // This should panic because index 5 is out of bounds for size 5 (indices 0..4)
            field.get_neuron(5);
        }
    }
}
