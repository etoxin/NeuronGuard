# Copyright 2026 Adam Lusted
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
InsectoidGait: High-level wrapper for the spatiotemporal ensemble mesh
and insectoid walking gait simulation.

Provides a clean API with named return types instead of raw tuples,
and convenience methods for common simulation patterns.
"""

from collections import namedtuple

GaitState = namedtuple("GaitState", [
    "step",
    "phases",
    "velocities",
    "active_nodes",
    "loop_intensities",
])
"""Snapshot of the insectoid simulation state at a single timestep."""

MutationEvent = namedtuple("MutationEvent", [
    "token_id",
    "evicted_target",
    "new_target",
    "timestamp_us",
])
"""A structural mutation event emitted by the Guard/Lease pattern."""


class InsectoidGait:
    """High-level wrapper for the insectoid rigid-body walking gait simulation.

    Wraps PyInsectoidSimulation with named return types and convenience methods
    for common patterns like perturbation injection and batch stepping.

    Example::

        sim = InsectoidGait()
        history = sim.run(steps=100)
        mutations = sim.perturb(target_leg=0)
        state = sim.step()
        print(state.phases, state.velocities)
    """

    def __init__(self):
        """Initialise the insectoid simulation.

        Automatically instantiates the underlying spatiotemporal mesh
        and configures the rhythmic CPG feedback loops.
        """
        from .neuronguard import PyInsectoidSimulation

        self._sim = PyInsectoidSimulation()
        self._step_count = 0

    def step(self, training_mode=False, correct_target=None, decay_factor=0.90):
        """Advance the simulation by one timestep.

        Args:
            training_mode: If True, performs topological plasticity.
            correct_target: Target index for training mode re-patching.
            decay_factor: Background decay multiplier (default 0.90).

        Returns:
            A GaitState namedtuple with the current simulation state.
        """
        self._sim.step(
            training_mode=training_mode,
            correct_target=correct_target,
            decay_factor=decay_factor,
        )
        self._step_count += 1

        return GaitState(
            step=self._step_count,
            phases=self._sim.get_phases(),
            velocities=self._sim.get_velocities(),
            active_nodes=self._sim.get_active_nodes(),
            loop_intensities=self._sim.get_loop_intensities(),
        )

    def run(self, steps, decay_factor=0.90):
        """Run the simulation for multiple steps and collect telemetry.

        Args:
            steps: Number of steps to run.
            decay_factor: Background decay multiplier.

        Returns:
            A list of GaitState namedtuples, one per step.
        """
        history = []
        for _ in range(steps):
            state = self.step(
                training_mode=False,
                correct_target=None,
                decay_factor=decay_factor,
            )
            history.append(state)
        return history

    def perturb(self, target_leg=0, decay_factor=0.90):
        """Inject a perturbation and return resulting structural mutations.

        Triggers a transactional structural re-patch (Guard/Lease) for the
        specified target leg. This simulates a sudden slip or push and
        measures the system's ability to self-stabilise.

        Args:
            target_leg: The leg index (0-5) to re-patch anomalous tokens to.
            decay_factor: Background decay multiplier.

        Returns:
            A list of MutationEvent namedtuples emitted during re-patching.
        """
        self._sim.step(
            training_mode=True,
            correct_target=target_leg,
            decay_factor=decay_factor,
        )
        self._step_count += 1

        raw_mutations = self._sim.get_structural_mutations()
        return [
            MutationEvent(
                token_id=m[0],
                evicted_target=m[1],
                new_target=m[2],
                timestamp_us=m[3],
            )
            for m in raw_mutations
        ]

    def get_mutations(self):
        """Drain and return any pending structural mutation events.

        Returns:
            A list of MutationEvent namedtuples.
        """
        raw = self._sim.get_structural_mutations()
        return [
            MutationEvent(
                token_id=m[0],
                evicted_target=m[1],
                new_target=m[2],
                timestamp_us=m[3],
            )
            for m in raw
        ]

    @property
    def phases(self):
        """Current phases of the 6 legs."""
        return self._sim.get_phases()

    @property
    def velocities(self):
        """Current velocities of the 6 legs."""
        return self._sim.get_velocities()

    @property
    def step_count(self):
        """Total number of steps executed."""
        return self._step_count
