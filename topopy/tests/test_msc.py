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
    topopy.MorseSmaleComplex
"""
from unittest import TestCase
import numpy as np
import topopy
from .testFunctions import gerber, generate_test_grid_2d
import sklearn
import os
import json


class TestMSC(TestCase):
    """
    Class for testing the Morse-Smale Complex
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
        self.msc = topopy.MorseSmaleComplex(debug=False)
        self.msc.build(self.X, self.Y)

    def test_debug(self):
        """ Test the debugging output of the MSC
        """
        self.setup()
        self.msc = topopy.MorseSmaleComplex(debug=True)
        self.msc.build(self.X, self.Y)

    def test_default(self):
        """ Test the build process of the MorseSmaleComplex
        """
        self.setup()

        self.assertEqual(len(self.msc.base_partitions.keys()), 16,
                         'The 2D Gerber test function should have 16 ' +
                         'partitions at the base level.')

    def test_get_classification(self):
        """ Testing the ability to retrieve the correct classification
            for a minimum, maximum, and regular data point
        """
        self.setup()

        self.assertEqual('minimum', self.msc.get_classification(0),
                         'get_classification has misidentified a local ' +
                         'minimum')
        self.assertEqual('maximum', self.msc.get_classification(429),
                         'get_classification has misidentified a local ' +
                         'maximum')
        self.assertEqual('regular', self.msc.get_classification(22),
                         'get_classification has misidentified a regular ' +
                         'point')

    def test_get_current_labels(self):
        """ Testing the ability of the MSC to report the currently
            active partition labels
        """
        self.setup()

        # TODO: These should be tuples like below. Di's attempt to
        # convert things to strings is causing conflicts
        self.assertSetEqual({'1599, 1189', '1584, 1170', '984, 410',
                             '999, 429', '999, 1189', '1584, 1189', '0, 410',
                             '960, 410', '24, 410', '960, 1170', '24, 429',
                             '984, 1170', '984, 429', '1560, 1170',
                             '984, 1189', '39, 429'},
                            set(self.msc.get_current_labels()), 'The base ' +
                            'partition labels returned from ' +
                            'get_current_labels does not match.')

        self.msc.set_persistence(self.msc.persistences[-1])
        self.assertSetEqual({(1599, 410)},
                            set(self.msc.get_current_labels()), 'The base ' +
                            'partition labels returned from ' +
                            'get_current_labels does not match.')

    def test_get_label(self):
        """ Testing the ability to retrieve the min-max label of any
            point in the data.
        """
        self.setup()

        self.assertEqual((0, 410), self.msc.get_label(0), 'The label of a ' +
                         'single index should be retrievable.')

        gold_labels = np.array([[0, 410], [0, 410]])
        test_labels = np.array(self.msc.get_label([0, 1]).flatten().tolist())
        np.testing.assert_array_equal(gold_labels, test_labels,
                                      'The label of a multiple indices ' +
                                      'should be retrievable.')

    def test_get_merge_sequence(self):
        """ Testing the ability to succinctly report the merge hierarchy
            of every extrema in the test data.
        """
        self.setup()

        self.assertDictEqual({0: (0.500192, 39, 10),
                              960: (0.117402, 1560, 1160),
                              1189: (0.117402, 429, 989),
                              39: (0.500192, 1599, 439),
                              999: (0.117402, 1599, 1199),
                              429: (0.256168, 410, 424),
                              1584: (0.256168, 1599, 1589),
                              1560: (0.500192, 1599, 1570),
                              1170: (0.117402, 410, 970),
                              24: (0.256168, 39, 29),
                              984: (0.117402, 1584, 1184),
                              410: (1.99362, 410, 410),
                              1599: (1.99362, 1599, 1599)},
                             self.msc.get_merge_sequence(),
                             'The merge sequence does not match the ' +
                             'expected output.')

    def test_get_weights(self):
        """ Function to test if default weights can be applied to the
            MorseSmaleComplex object. This feature should be evaluated
            in order to determine if it makes sense to keep it.
        """
        self.setup()

        equal_weights = np.ones(len(self.Y))*1.0/float(len(self.Y))

        np.testing.assert_array_equal(equal_weights,
                                      self.msc.get_weights(),
                                      'The default weights should be 1 for ' +
                                      'every row in the input.')

        test_weights = self.msc.get_weights(list(range(len(self.msc.w))))

        np.testing.assert_array_equal(equal_weights, test_weights,
                                      'User should be able to filter the ' +
                                      'rows retrieved from get_weights.')

    def test_load_data_and_build(self):
        """ Tests that loading the same test data from file yields an
            equivalent result
        """
        self.setup()

        msc = topopy.MorseSmaleComplex()

        np.savetxt("test_file.csv", np.hstack((self.X,
                                               np.atleast_2d(self.Y).T)),
                   delimiter=",", header=','.join(self.msc.names))
        msc.load_data_and_build('test_file.csv', delimiter=',')
        os.remove('test_file.csv')

        self.assertListEqual(self.msc.hierarchy, msc.hierarchy, 'loading ' +
                             'from file should produce the same hierarchy')
        self.assertDictEqual(self.msc.base_partitions, msc.base_partitions,
                             'loading from file should produce the base ' +
                             'partitions.')

    def test_persistence(self):
        """ Tests the getting and setting of different persistence
            values. Will also test that the number of components
            decreases correctly as persistence is increased
        """
        self.setup()
        self.msc.set_persistence(self.msc.persistences[1])
        self.assertEqual(self.msc.persistences[1], self.msc.get_persistence(),
                         'Users should be able to get and set the ' +
                         'persistence.')

    def test_print_hierarchy(self):
        """ Testing the printing of the MSC hierarchy
        """
        self.setup()
        self.assertEqual('Minima,0.500192,0,39,10 Minima,0.256168,24,39,29 ' +
                         'Minima,0.500192,39,1599,439 Minima,0.117402,960,' +
                         '1560,1160 Minima,0.117402,984,1584,1184 Minima,' +
                         '0.117402,999,1599,1199 Minima,0.500192,1560,1599,' +
                         '1570 Minima,0.256168,1584,1599,1589 Minima,' +
                         '1.99362,1599,1599,1599 Maxima,1.99362,410,410,410 ' +
                         'Maxima,0.256168,429,410,424 Maxima,0.117402,1170,' +
                         '410,970 Maxima,0.117402,1189,429,989 ',
                         self.msc.print_hierarchy(), 'The hierarchy printed ' +
                         'differs from what is expected.')

    def test_reset(self):
        """ Tests resetting of internal storage of the msc object
        """
        self.setup()
        self.msc.reset()

        self.assertListEqual([], self.msc.persistences, 'reset should clear ' +
                             'all internal storage of the msc.')
        self.assertDictEqual({}, self.msc.partitions, 'reset should clear ' +
                             'all internal storage of the msc.')
        self.assertDictEqual({}, self.msc.base_partitions, 'reset should ' +
                             'clear all internal storage of the msc.')
        self.assertEqual(0, self.msc.persistence, 'reset should ' +
                         'clear all internal storage of the msc.')
        self.assertDictEqual({}, self.msc.mergeSequence, 'reset should ' +
                             'clear all internal storage of the msc.')
        self.assertListEqual([], self.msc.minIdxs, 'reset should clear all ' +
                             'internal storage of the msc.')
        self.assertListEqual([], self.msc.maxIdxs, 'reset should clear all ' +
                             'internal storage of the msc.')
        self.assertEqual([], self.msc.X, 'reset should clear all ' +
                         'internal storage of the msc.')
        self.assertEqual([], self.msc.Y, 'reset should clear all ' +
                         'internal storage of the msc.')
        self.assertEqual([], self.msc.w, 'reset should clear all ' +
                         'internal storage of the msc.')
        self.assertEqual([], self.msc.names, 'reset should clear all ' +
                         'internal storage of the msc.')
        self.assertEqual([], self.msc.Xnorm, 'reset should clear all ' +
                         'internal storage of the msc.')
        self.assertEqual(None, self.msc.hierarchy, 'reset should ' +
                         'clear all internal storage of the msc.')

    def test_save(self):
        """ Testing the save feature correctly dumps to a json file.
        """
        self.setup()
        self.msc.save('test.csv', 'test.json')

        with open('gold.csv', 'r') as data_file:
            gold_csv = data_file.read()
        with open('gold.json', 'r') as data_file:
            gold_json = data_file.read()
            gold_json = json.loads(gold_json)

        with open('test.csv', 'r') as data_file:
            test_csv = data_file.read()
        with open('test.json', 'r') as data_file:
            test_json = data_file.read()
            test_json = json.loads(test_json)

        self.assertDictEqual(gold_json, test_json,
                             'save does not reproduce the same results ' +
                             'as the gold custom json file.')
        self.assertMultiLineEqual(gold_csv, test_csv,
                                  'save does not reproduce the same results ' +
                                  'as the gold custom csv file.')
        os.remove('test.json')
        os.remove('test.csv')

        self.msc.save()
        with open('Base_Partition.json', 'r') as data_file:
            test_json = data_file.read()
            test_json = json.loads(test_json)
        with open('Hierarchy.csv', 'r') as data_file:
            test_csv = data_file.read()

        self.assertDictEqual(gold_json, test_json,
                             'save does not reproduce the same results ' +
                             'as the gold default json file.')
        self.assertMultiLineEqual(gold_csv, test_csv,
                                  'save does not reproduce the same results ' +
                                  'as the gold default csv file.')
        os.remove('Base_Partition.json')
        os.remove('Hierarchy.csv')

    def test_shape_functions(self):
        """ Test the get_dimensionality and get_sample_size functions
        """
        self.setup()

        self.assertEqual(self.X.shape[1], self.msc.get_dimensionality(),
                         'get_dimensionality should return the number of ' +
                         'columns in X.')
        self.assertEqual(self.X.shape[0], self.msc.get_sample_size(),
                         'get_sample_size should return the number of ' +
                         'rows in X.')
        self.assertEqual(121, self.msc.get_sample_size('1599, 1189'),
                         'get_sample_size should return the number of ' +
                         'rows in the specified partition.')

# get_partitions
# get_stable_manifolds
# get_unstable_manifolds
