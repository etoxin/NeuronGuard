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

pub const NUM_MACRO_CLUSTERS: usize = 50;
pub const TOKENS_PER_CLUSTER: usize = 1000;
pub const TOTAL_TOKENS: usize = NUM_MACRO_CLUSTERS * TOKENS_PER_CLUSTER;

/// HierarchicalWinnerTakeAll
/// Implements a two-tier cascading search to find the winning token ID.
/// This prevents L1/L2 cache thrashing by isolating the search to a highly constrained
/// grammatical cluster before activating a hyper-specific, cache-resident memory block.
pub struct HierarchicalWinnerTakeAll {
    pub macro_potentials: [i32; NUM_MACRO_CLUSTERS],
    // Micro potentials are stored in a flat array of size 50,000,
    // but we only access the 1,000 elements corresponding to the winning macro cluster.
    pub micro_potentials: Vec<i32>,
}

impl HierarchicalWinnerTakeAll {
    /// Creates a new Hierarchical WTA selector.
    pub fn new() -> Self {
        Self {
            macro_potentials: [0; NUM_MACRO_CLUSTERS],
            micro_potentials: vec![0; TOTAL_TOKENS],
        }
    }

    /// Resets all potentials to zero.
    pub fn reset(&mut self) {
        self.macro_potentials = [0; NUM_MACRO_CLUSTERS];
        self.micro_potentials.fill(0);
    }

    /// Tier-1 (Macro Sieve): Find the macro cluster with the highest potential.
    /// Spikes isolate the target context down to a highly constrained grammatical cluster.
    pub fn select_macro_cluster(&self) -> usize {
        let mut winning_cluster = 0;
        let mut max_potential = i32::MIN;
        for i in 0..NUM_MACRO_CLUSTERS {
            if self.macro_potentials[i] > max_potential {
                max_potential = self.macro_potentials[i];
                winning_cluster = i;
            }
        }
        winning_cluster
    }

    /// Tier-2 (Micro Target): Activates a hyper-specific, 64-byte aligned memory block
    /// containing the final target token indices for the winning macro cluster, and finds the winning token.
    pub fn select_winning_token(&self, winning_cluster: usize) -> u32 {
        let start_idx = winning_cluster * TOKENS_PER_CLUSTER;
        let end_idx = start_idx + TOKENS_PER_CLUSTER;

        let mut winning_token_offset = 0;
        let mut max_potential = i32::MIN;

        // Only search the 1,000 tokens in the winning cluster's memory block!
        // This fits perfectly in a standard L1/L2 cache line and avoids thrashing.
        for i in start_idx..end_idx {
            if self.micro_potentials[i] > max_potential {
                max_potential = self.micro_potentials[i];
                winning_token_offset = i - start_idx;
            }
        }

        (start_idx + winning_token_offset) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_wta() {
        let mut wta = HierarchicalWinnerTakeAll::new();

        // Accumulate potentials in macro cluster 3 and 7
        wta.macro_potentials[3] = 100;
        wta.macro_potentials[7] = 250; // Winning macro cluster

        // Accumulate potentials in micro targets for cluster 7
        let start_7 = 7 * TOKENS_PER_CLUSTER;
        wta.micro_potentials[start_7 + 42] = 50;
        wta.micro_potentials[start_7 + 108] = 120; // Winning token in cluster 7
        wta.micro_potentials[start_7 + 512] = 30;

        // Run Tier-1 Sieve
        let winning_cluster = wta.select_macro_cluster();
        assert_eq!(winning_cluster, 7);

        // Run Tier-2 Target Selector
        let winning_token = wta.select_winning_token(winning_cluster);
        assert_eq!(winning_token, (start_7 + 108) as u32);
    }
}
