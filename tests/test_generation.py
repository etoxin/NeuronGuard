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

import unittest

from neuronguard import NeuronGuardTokenizer, run_autoregressive_generation


class TestNeuronGuardGen(unittest.TestCase):
    def setUp(self):
        self.tokenizer = NeuronGuardTokenizer()

    def test_tokenizer_encode_decode(self):
        text = "the and ing ion ent for that tis es en to it is was he she his her in on at by an with this you not but or as"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(text, decoded)

    def test_topological_fields_split(self):
        text = "The quick brown Fox jumps over the lazy Dog."
        field_0, field_1, field_2 = self.tokenizer.split_topological_fields(text)

        # All fields should have the exact same length
        self.assertEqual(len(field_0), len(field_1))
        self.assertEqual(len(field_0), len(field_2))

        # Field 0 should contain valid token IDs
        self.assertTrue(all(0 <= tid < 50000 for tid in field_0))

        # Field 1 should contain structural indicators (0 to 3)
        self.assertTrue(all(0 <= val <= 3 for val in field_1))

        # Field 2 should contain distance profiles (0 to 255)
        self.assertTrue(all(0 <= val <= 255 for val in field_2))

    def test_autoregressive_generation(self):
        prompt = "The quick brown Fox"
        prompt_ids = self.tokenizer.encode(prompt)

        # Run generation
        generated = run_autoregressive_generation(
            prompt_ids, max_generation_length=10, temperature=0.7
        )

        # Verify output
        self.assertIsInstance(generated, list)
        self.assertTrue(len(generated) <= 10)
        self.assertTrue(all(isinstance(tid, int) for tid in generated))

        # Decode output
        decoded = self.tokenizer.decode(generated)
        self.assertIsInstance(decoded, str)

    def test_ffi_bindings(self):
        import neuronguard as ng

        # Test PyPermanentNeuromorphicLine
        line = ng.PyPermanentNeuromorphicLine()
        line.synapses_weights = [1, 2, 3, 4, 5, 6, 7, 8] + [0] * 16
        self.assertEqual(line.synapses_weights[:8], [1, 2, 3, 4, 5, 6, 7, 8])

        line.target_ids = [10, 20, 30] + [0] * 21
        self.assertEqual(line.target_ids[:3], [10, 20, 30])

        # Test PySpikingAttentionState
        state = ng.PySpikingAttentionState()
        state.reset()
        state.update([1, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8])
        res = state.query([1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(res, [1, 2, 3, 4, 5, 6, 7, 8])

        # Test PyHierarchicalWinnerTakeAll
        wta = ng.PyHierarchicalWinnerTakeAll()
        wta.reset()
        potentials = [0] * 50
        potentials[5] = 100
        wta.macro_potentials = potentials
        self.assertEqual(wta.select_macro_cluster(), 5)

    def test_trainer_field(self):
        import os

        import neuronguard as ng

        trainer = ng.NeuronGuardTrainerField(sensory_count=10, motor_count=10)
        trainer.reset_potentials()
        trainer.train_stream_step_sync([2, 5])

        # Test serialization to base64 file
        path = "test_weights.txt"
        trainer.save_weights_to_b64(path)
        self.assertTrue(os.path.exists(path))

        # Clean up
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
