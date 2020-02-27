"""TopoPy - Topological constructs for Python.

TopoPy is a Python package for constructing approximate topological constructs
in arbitrary dimensions using a neighborhood graph structure for approximating
local gradient.

"""

from __future__ import absolute_import

# These lines ensure that we do not have to do something like:
# 'from contrib.MorseSmaleComplex import MorseSmaleComplex' outside of
# this submodule
from .TopologicalObject import TopologicalObject
from .MorseComplex import MorseComplex
from .MorseSmaleComplex import MorseSmaleComplex
from .MergeTree import MergeTree
from .ContourTree import ContourTree

__all__ = [
    "TopologicalObject",
    "MorseComplex",
    "MorseSmaleComplex",
    "MergeTree",
    "ContourTree",
]
__version__ = "1.0.1"
