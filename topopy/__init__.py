"""
The Topology module includes the approximate Morse-Smale complex
(MorseSmaleComplex) and Merge Tree computation.

Created on January 11, 2016
@author: maljovec
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
__version__ = "0.1.1"
