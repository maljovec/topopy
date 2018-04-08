########################################################################
# Software License Agreement (BSD License)                             #
#                                                                      #
# Copyright 2018 University of Utah                                    #
# Scientific Computing and Imaging Institute                           #
# 72 S Central Campus Drive, Room 3750                                 #
# Salt Lake City, UT 84112                                             #
#                                                                      #
# THE BSD LICENSE                                                      #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# 1. Redistributions of source code must retain the above copyright    #
#    notice, this list of conditions and the following disclaimer.     #
# 2. Redistributions in binary form must reproduce the above copyright #
#    notice, this list of conditions and the following disclaimer in   #
#    the documentation and/or other materials provided with the        #
#    distribution.                                                     #
# 3. Neither the name of the copyright holder nor the names of its     #
#    contributors may be used to endorse or promote products derived   #
#    from this software without specific prior written permission.     #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR #
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED       #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE   #
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY       #
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL   #
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE    #
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS        #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER #
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR      #
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN  #
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                        #
########################################################################
"""
    This module will test the basic functionalities of
    topopy.ContourTree and by proxy the topopy.MergeTree
"""
from unittest import TestCase
import numpy as np
import topopy
from .testFunctions import gerber, generate_test_grid_2d
import sklearn


class TestCT(TestCase):
    """
    Class for testing the Contour Tree and its prerequisite the Merge Tree
    """

    def setup(self):
        """ Setup function will create a fixed point set and parameter
            settings for testing different aspects of this library.
        """
        self.X = generate_test_grid_2d(40)
        self.Y = gerber(self.X)

        self.norm_x = {}
        scaler = sklearn.preprocessing.MinMaxScaler()
        self.norm_x['feature'] = scaler.fit_transform(np.atleast_2d(self.X))
        self.norm_x['zscore'] = sklearn.preprocessing.scale(self.X, axis=0,
                                                            with_mean=True,
                                                            with_std=True,
                                                            copy=True)
        self.norm_x['none'] = self.X

        # Methods covered here:
        # __init__
        # build
        # __set_data
        self.ct = topopy.ContourTree(debug=False)
        self.ct.build(self.X, self.Y)

    def test_debug(self):
        """ Test the debugging output of the CT
        """
        self.setup()
        self.ct = topopy.ContourTree(debug=True)
        self.ct.build(self.X, self.Y)

    def test_default(self):
        """ Test the build process of the ContourTree
        """
        self.setup()

        self.assertEqual(24, len(self.ct.superNodes),
                         'The 2D Gerber test function should have 24 ' +
                         'nodes in its contour tree.')
        self.assertEqual(23, len(self.ct.superArcs),
                         'The 2D Gerber test function should have 23 ' +
                         'arcs in its contour tree.')

    def test_no_short_circuit(self):
        """ Test the build process of the ContourTree
        """
        self.setup()
        self.ct = topopy.ContourTree(short_circuit=False)
        self.ct.build(self.X, self.Y)

        self.assertEqual(71, len(self.ct.superNodes),
                         'The 2D Gerber test function should have 71 ' +
                         'nodes in its contour tree.')
        self.assertEqual(70, len(self.ct.superArcs),
                         'The 2D Gerber test function should have 70 ' +
                         'arcs in its contour tree.')

    def test_get_seeds(self):
        """ Test the build process of the ContourTree
        """
        self.setup()
        seeds = self.ct.get_seeds(0, False)

        self.assertEqual(24, len(self.ct.superNodes),
                         'The 2D Gerber test function should have 24 ' +
                         'nodes in its contour tree.')
        self.assertEqual(23, len(self.ct.superArcs),
                         'The 2D Gerber test function should have 23 ' +
                         'arcs in its contour tree.')

        seeds = self.ct.get_seeds(0, True)

    # def test_persistence(self):
    #     """ Tests the getting and setting of different persistence
    #         values. Will also test that the number of components
    #         decreases correctly as persistence is increased
    #     """
    #     self.setup()
    #     self.ct.set_persistence(self.ct.persistences[1])
    #     self.assertEqual(self.ct.persistences[1], self.ct.get_persistence(),
    #                      'Users should be able to get and set the ' +
    #                      'persistence.')
