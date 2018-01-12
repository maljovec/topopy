"""
The Topology module includes the approximate Morse-Smale complex
(MorseSmaleComplex) code, all of its associated visualization views and the
MergeTree computation.

Created on January 11, 2016
@author: maljdp
"""

from __future__ import absolute_import

## These lines ensure that we do not have to do something like:
## 'from contrib.MorseSmaleComplex import MorseSmaleComplex' outside of this
## submodule
from .MorseSmaleComplex import MorseSmaleComplex
from .MergeTree import MergeTree
from .ContourTree import ContourTree
# from .MainWindow import MainWindow

# We should not really need this as we do not use wildcard imports
__all__ = ['MorseSmaleComplex','MergeTree','ContourTree']
