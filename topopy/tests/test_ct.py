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
from .testFunctions import gerber
import sklearn
import os
import json


class TestCT(TestCase):
    """
    Class for testing the Morse-Smale Complex
    """

    def setup(self):
        """ Setup function will create a fixed point set and parameter
            settings for testing different aspects of this library.
        """
        x, y = np.mgrid[0:1:40j, 0:1:40j]
        self.X = np.vstack([x.ravel(), y.ravel()]).T
        self.Y = gerber(self.X)

        self.norm_x = {}
        scaler = sklearn.preprocessing.MinMaxScaler()
        self.norm_x['feature'] = scaler.fit_transform(np.atleast_2d(self.X))
        self.norm_x['zscore'] = sklearn.preprocessing.scale(self.X, axis=0,
                                                            with_mean=True,
                                                            with_std=True,
                                                            copy=True)
        self.norm_x['none'] = self.X

        # TODO: convert CT to use this type of API like MSC:
        # Methods covered here:
        # __init__
        # build
        # __set_data
        self.ct = topopy.ContourTree(self.X, self.Y, debug=False)

    def test_debug(self):
        """ Test the debugging output of the CT
        """
        self.setup()
        self.ct = topopy.ContourTree(self.X, self.Y, debug=True)

    def test_default(self):
        """ Test the build process of the MorseSmaleComplex
        """
        self.setup()

        self.assertEqual(len(self.ct.sortedNodes), 16,
                         'The 2D Gerber test function should have 16 ' +
                         'nodes at the base level.')

    def test_get_names(self):
        """ Test the ability for the code to generate dummy names
            and also correctly use passed in names.
        """
        self.setup()

        default_names = []
        for d in range(self.ct.get_dimensionality()):
            default_names.append('x%d' % d)
        default_names.append('y')

        test_names = self.ct.get_names()
        for i in range(len(default_names)):
            self.assertEqual(default_names[i], test_names[i],
                             'The MorseSmaleComplex object should generate ' +
                             'default value names for labeling purposes.')

        custom_names = ['a', 'b', 'c']
        self.ct.build(self.X, self.Y, names=custom_names)
        test_names = self.ct.get_names()
        for i in range(len(custom_names)):
            self.assertEqual(custom_names[i], test_names[i],
                             'The MorseSmaleComplex object should use any ' +
                             'custom names passed in for labeling purposes.')

    def test_get_normed_x(self):
        """ Tests get_normed_x in several different contexts:
                Single Element extraction
                Single Column extraction
                Single Row extraction
                Multiple row extraction
                Multiple column extraction
                Full data extraction
        """
        self.setup()

        for norm, X in self.norm_x.items():
            ct = topopy.MorseSmaleComplex(normalization=norm)
            ct.build(self.X, self.Y)

            # Test single column extraction
            for col in range(X.shape[1]):
                column_values = ct.get_normed_x(cols=col)
                np.testing.assert_array_equal(X[:, col], column_values,
                                              'get_normed_x should be able ' +
                                              'to access a full column of ' +
                                              'the input data.')

            # Test single row extraction
            for row in range(X.shape[0]):
                row_values = ct.get_normed_x(row).flatten()
                np.testing.assert_array_equal(X[row, :], row_values,
                                              'get_normed_x should be able ' +
                                              'to access a full row of the ' +
                                              'input data.')
                # Test single element extraction
                for col in range(X.shape[1]):
                    self.assertEqual(X[row, col],
                                     ct.get_normed_x(row, col),
                                     'get_normed_x should be able to access ' +
                                     'a single element of the input data.')

            # Multiple row extraction
            row_values = ct.get_normed_x(list(range(X.shape[0])), 0)
            np.testing.assert_array_equal(X[:, 0], row_values,
                                          'get_normed_x should be able to ' +
                                          'access multiple rows of the ' +
                                          'input data.')

            # Multiple column extraction
            col_values = ct.get_normed_x(0, list(range(X.shape[1]))).flatten()
            np.testing.assert_array_equal(X[0, :], col_values,
                                          'get_normed_x should be able to ' +
                                          'access multiple columns of the ' +
                                          'input data.')

            # Full data extraction
            np.testing.assert_array_equal(X, ct.get_normed_x(),
                                          'get_normed_x should be able to ' +
                                          'access the entire input data.')

    def test_get_weights(self):
        """ Function to test if default weights can be applied to the
            MorseSmaleComplex object. This feature should be evaluated
            in order to determine if it makes sense to keep it.
        """
        self.setup()

        equal_weights = np.ones(len(self.Y))*1.0/float(len(self.Y))

        np.testing.assert_array_equal(equal_weights,
                                      self.ct.get_weights(),
                                      'The default weights should be 1 for ' +
                                      'every row in the input.')

        test_weights = self.ct.get_weights(list(range(len(self.ct.w))))

        np.testing.assert_array_equal(equal_weights, test_weights,
                                      'User should be able to filter the ' +
                                      'rows retrieved from get_weights.')

    def test_get_x(self):
        """ Tests get_x in several different contexts:
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
            column_values = self.ct.get_x(cols=col)
            np.testing.assert_array_equal(self.X[:, col], column_values,
                                          'get_x should be able to access a ' +
                                          'full column of the input data.')

        # Test single row extraction
        for row in range(self.X.shape[0]):
            row_values = self.ct.get_x(row).flatten()
            np.testing.assert_array_equal(self.X[row, :], row_values,
                                          'get_x should be able to access a ' +
                                          'full row of the input data.')
            # Test single element extraction
            for col in range(self.X.shape[1]):
                self.assertEqual(self.X[row, col], self.ct.get_x(row, col),
                                 'get_x should be able to access a single ' +
                                 'element of the input data.')

        # Multiple row extraction
        row_values = self.ct.get_x(list(range(self.X.shape[0])), 0)
        np.testing.assert_array_equal(self.X[:, 0], row_values,
                                      'get_x should be able to access ' +
                                      'multiple rows of the input data.')

        # Multiple column extraction
        col_values = self.ct.get_x(0, list(range(self.X.shape[1]))).flatten()
        np.testing.assert_array_equal(self.X[0, :], col_values,
                                      'get_x should be able to access ' +
                                      'multiple columns of the input data.')

        # Full data extraction
        np.testing.assert_array_equal(self.X, self.ct.get_x(),
                                      'get_x should be able to access ' +
                                      'the entire input data.')

        # Empty query
        np.testing.assert_array_equal([], self.ct.get_x([]),
                                      'get_x should be able to access ' +
                                      'return an empty array on null filter.')

    def test_get_y(self):
        """ Tests get_y in several different contexts:
                Single Element extraction
                Multiple row extraction
                Full data extraction
        """
        self.setup()

        for row in range(len(self.Y)):
            # Test single element extraction
            self.assertEqual(self.Y[row], self.ct.get_y(row),
                             'get_y should be able to access a single ' +
                             'element of the input data.')

        # Multiple row extraction
        row_values = self.ct.get_y(list(range(len(self.Y))))
        np.testing.assert_array_equal(self.Y, row_values,
                                      'get_y should be able to access ' +
                                      'multiple rows of the input data.')

        # Full data extraction
        np.testing.assert_array_equal(self.Y, self.ct.get_y(),
                                      'get_y should be able to access ' +
                                      'the entire input data.')

        # Empty query
        np.testing.assert_array_equal([], self.ct.get_y([]),
                                      'get_y should be able to access ' +
                                      'return an empty array on null filter.')

    def test_neighbors(self):
        """ Tests the ability to retrieve the neighbors of a given index
        """
        self.setup()
        self.assertSetEqual({40, 1}, set(self.ct.get_neighbors(0)),
                            'get_neighbors should return a list of integer ' +
                            'indices indicating who is connected to the ' +
                            'given index.')

    def test_persistence(self):
        """ Tests the getting and setting of different persistence
            values. Will also test that the number of components
            decreases correctly as persistence is increased
        """
        self.setup()
        self.ct.set_persistence(self.ct.persistences[1])
        self.assertEqual(self.ct.persistences[1], self.ct.get_persistence(),
                         'Users should be able to get and set the ' +
                         'persistence.')

    # def test_reset(self):
    #     """ Tests resetting of internal storage of the ct object
    #     """
    #     self.setup()
    #     self.ct.reset()

    #     self.assertListEqual([], self.ct.persistences, 'reset should clear ' +
    #                          'all internal storage of the ct.')
    #     self.assertDictEqual({}, self.ct.partitions, 'reset should clear ' +
    #                          'all internal storage of the ct.')
    #     self.assertDictEqual({}, self.ct.base_partitions, 'reset should ' +
    #                          'clear all internal storage of the ct.')
    #     self.assertEqual(0, self.ct.persistence, 'reset should ' +
    #                      'clear all internal storage of the ct.')
    #     self.assertDictEqual({}, self.ct.mergeSequence, 'reset should ' +
    #                          'clear all internal storage of the ct.')
    #     self.assertListEqual([], self.ct.minIdxs, 'reset should clear all ' +
    #                          'internal storage of the ct.')
    #     self.assertListEqual([], self.ct.maxIdxs, 'reset should clear all ' +
    #                          'internal storage of the ct.')
    #     self.assertEqual([], self.ct.X, 'reset should clear all ' +
    #                      'internal storage of the ct.')
    #     self.assertEqual([], self.ct.Y, 'reset should clear all ' +
    #                      'internal storage of the ct.')
    #     self.assertEqual([], self.ct.names, 'reset should clear all ' +
    #                      'internal storage of the ct.')
    #     self.assertEqual([], self.ct.Xnorm, 'reset should clear all ' +
    #                      'internal storage of the ct.')
    #     self.assertEqual(None, self.ct.hierarchy, 'reset should ' +
    #                      'clear all internal storage of the ct.')

    def test_shape_functions(self):
        """ Test the get_dimensionality and get_sample_size functions
        """
        self.setup()

        self.assertEqual(self.X.shape[1], self.ct.get_dimensionality(),
                         'get_dimensionality should return the number of ' +
                         'columns in X.')
        self.assertEqual(self.X.shape[0], self.ct.get_sample_size(),
                         'get_sample_size should return the number of ' +
                         'rows in X.')
        self.assertEqual(121, self.ct.get_sample_size('1599, 1189'),
                         'get_sample_size should return the number of ' +
                         'rows in the specified partition.')
