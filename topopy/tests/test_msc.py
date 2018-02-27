##############################################################################
# Software License Agreement (BSD License)                                   #
#                                                                            #
# Copyright 2018 University of Utah                                          #
# Scientific Computing and Imaging Institute                                 #
# 72 S Central Campus Drive, Room 3750                                       #
# Salt Lake City, UT 84112                                                   #
#                                                                            #
# THE BSD LICENSE                                                            #
#                                                                            #
# Redistribution and use in source and binary forms, with or without         #
# modification, are permitted provided that the following conditions         #
# are met:                                                                   #
#                                                                            #
# 1. Redistributions of source code must retain the above copyright          #
#    notice, this list of conditions and the following disclaimer.           #
# 2. Redistributions in binary form must reproduce the above copyright       #
#    notice, this list of conditions and the following disclaimer in the     #
#    documentation and/or other materials provided with the distribution.    #
# 3. Neither the name of the copyright holder nor the names of its           #
#    contributors may be used to endorse or promote products derived         #
#    from this software without specific prior written permission.           #
#                                                                            #
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       #
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  #
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    #
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   #
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  #
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      #
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   #
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          #
##############################################################################
"""
    This module will test the basic functionalities of
    topopy.MorseSmaleComplex
"""
from unittest import TestCase
import numpy as np
import topopy
from .testFunctions import gerber


class TestMSC(TestCase):
    """
    Class for testing the Morse-Smale Complex
    """

    def setup(self):
        """
        Setup function will create a fixed point set and parameter
        settings for testing different aspects of this library.
        """
        x, y = np.mgrid[0:1:40j, 0:1:40j]
        self.X = np.vstack([x.ravel(), y.ravel()]).T
        self.Y = gerber(self.X)

        # Methods covered here:
        # __init__
        # build
        self.msc = topopy.MorseSmaleComplex(debug=True)
        self.msc.build(self.X, self.Y)

    def test_get_x(self):
        """
        Tests get_x in several different contexts:
            Single Element extraction
            Single Column extraction
            Single Row extraction
            Multiple row extraction
            Multiple column extraction
            Full data extraction
        """
        self.setup()

        # Test single column extraction
        for col in range(self.X.shape[1]):
            column_values = self.msc.get_x(cols=col)
            np.testing.assert_array_equal(self.X[:, col], column_values,
                                          'get_x should be able to access a ' +
                                          'full column of the input data.')

        # Test single row extraction
        for row in range(self.X.shape[0]):
            row_values = self.msc.get_x(row).flatten()
            np.testing.assert_array_equal(self.X[row, :], row_values,
                                          'get_x should be able to access a ' +
                                          'full row of the input data.')
            # Test single element extraction
            for col in range(self.X.shape[1]):
                self.assertEqual(self.X[row, col], self.msc.get_x(row, col),
                                 'get_x should be able to access a single ' +
                                 'element of the input data.')

        # Multiple row extraction
        row_values = self.msc.get_x(list(range(self.X.shape[0])), 0)
        np.testing.assert_array_equal(self.X[:, 0], row_values,
                                      'get_x should be able to access ' +
                                      'multiple rows of the input data.')

        # Multiple column extraction
        col_values = self.msc.get_x(0, list(range(self.X.shape[1]))).flatten()
        np.testing.assert_array_equal(self.X[0, :], col_values,
                                      'get_x should be able to access ' +
                                      'multiple columns of the input data.')

        # Full data extraction
        np.testing.assert_array_equal(self.X, self.msc.get_x(),
                                      'get_x should be able to access ' +
                                      'the entire input data.')

    def test_default(self):
        """
        Blank function serving as a template
        """
        self.setup()

        self.assertEqual(len(self.msc.base_partitions.keys()), 16,
                         'The 2D Gerber test function should have 16 ' +
                         'partitions at the base level.')

# get_classification
# get_current_labels
# get_dimensionality
# get_label
# get_merge_sequence
# get_names
# get_neighbors
# get_normed_x
# get_partitions
# get_persistence
# get_sample_size
# get_stable_manifolds
# get_unstable_manifolds
# get_weights
# get_x
# get_y

# load_data_and_build
# print_hierarchy
# save
# set_persistence
# set_weights
# reset