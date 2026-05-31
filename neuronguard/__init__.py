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
NeuronGuard: Cache-Aligned Neuromorphic Event Engine

This package provides both the raw Rust bindings and a high-level Python SDK.

Raw bindings (backward compatible)::

    import neuronguard as ng
    field = ng.NeuronGuardField(sensory_count=1000, motor_count=4)

High-level SDK::

    from neuronguard import TextClassifier, TabularClassifier
"""

# Re-export the compiled Rust extension classes at the top level
# for full backward compatibility with existing code.
# The compiled .so file is placed in this directory by maturin.
try:
    from .neuronguard import (  # noqa: F401
        NeuronGuardField,
    )
except ImportError:
    # Allow import during type checking or docs build when the
    # Rust extension hasn't been compiled yet.
    pass

# High-level SDK exports

from .tabular import TabularClassifier  # noqa: F401
from .text import TextClassifier  # noqa: F401
from .tokenizer import DEFAULT_STOP_WORDS, tokenize  # noqa: F401
from .vocab import build_vocab  # noqa: F401

__version__ = "0.2.0"
