"""
    This module will test the basic functionalities of
    topopy.MorseComplex
"""
from unittest import TestCase
import numpy as np
import topopy
from .test_functions import gerber, generate_test_grid_2d
import sklearn
import json
import sys
import os


class TestMC(TestCase):
    """
    Class for testing the Morse Complex
    """

    def setup(self):
        """
        Setup function will create a fixed point set and parameter
        settings for testing different aspects of this library.
        """
        self.X = generate_test_grid_2d(40)
        self.Y = gerber(self.X)

        self.norm_x = {}
        scaler = sklearn.preprocessing.MinMaxScaler()
        self.norm_x["feature"] = scaler.fit_transform(np.atleast_2d(self.X))
        self.norm_x["zscore"] = sklearn.preprocessing.scale(
            self.X, axis=0, with_mean=True, with_std=True, copy=True
        )
        self.norm_x["none"] = self.X

        # Methods covered here:
        # __init__
        # build
        # __set_data
        self.test_object = topopy.MorseComplex(debug=False, max_neighbors=10)
        self.test_object.build(self.X, self.Y)

    def test_debug(self):
        """
        Test the debugging output of the MorseComplex
        """
        self.setup()

        test_file = 'mc_test_debug.txt'
        sys.stdout = open(test_file, 'w')

        self.test_object = topopy.MorseComplex(debug=True, max_neighbors=10)
        self.test_object.build(self.X, self.Y)

        sys.stdout.close()

        lines = ["Graph Preparation:",
                 "Decomposition:"]

        with open(test_file, 'r') as fp:
            debug_output = fp.read()
            for line in lines:
                self.assertIn(line, debug_output)

        os.remove(test_file)

        # Restore stdout
        sys.stdout = sys.__stdout__

    def test_default(self):
        """
        Test the build process of the MorseComplex
        """
        self.setup()

        self.assertEqual(
            len(self.test_object.base_partitions.keys()),
            4,
            "The 2D Gerber test function should have 4 stable manifolds at the base level.",
        )

    def test_get_classification(self):
        """
        Testing the ability to retrieve the correct classification
        for a minimum, maximum, and regular data point
        """
        self.setup()

        self.assertEqual(
            "maximum",
            self.test_object.get_classification(429),
            "get_classification has misidentified a local maximum",
        )
        self.assertEqual(
            "regular",
            self.test_object.get_classification(22),
            "get_classification has misidentified a regular point",
        )

    def test_get_current_labels(self):
        """
        Testing the ability of the MSC to report the currently
        active partition labels
        """
        self.setup()

        self.assertSetEqual(
            {
                1189,
                1170,
                410,
                429
            },
            set(self.test_object.get_current_labels()),
            "The base partition labels returned from get_current_labels does not match.",
        )

        self.test_object.set_persistence(self.test_object.persistences[-1])
        self.assertSetEqual(
            {410},
            set(self.test_object.get_current_labels()),
            "The base partition labels returned from get_current_labels does not match.",
        )

    def test_get_label(self):
        """
        Testing the ability to retrieve the min-max label of any
        point in the data.
        """
        self.setup()

        self.assertEqual(
            410,
            self.test_object.get_label(0),
            "The label of a single index should be retrievable.",
        )

        self.assertEqual(
            len(self.Y),
            len(self.test_object.get_label()),
            "The label of the entire dataset should be retrievable.",
        )

        gold_labels = np.array([410,410])
        test_labels = np.array(self.test_object.get_label([0, 1]).flatten().tolist())
        np.testing.assert_array_equal(
            gold_labels,
            test_labels,
            "The label of a multiple indices should be retrievable.",
        )

        self.assertEqual(
            len(self.Y),
            len(self.test_object.get_label()),
            "Requesting labels without specifying an index should return a list of all labels",
        )

        self.assertEqual(
            [],
            self.test_object.get_label([]),
            "Requesting labels for an empty query should return an empty list",
        )

    def test_get_merge_sequence(self):
        """
        Testing the ability to succinctly report the merge hierarchy
        of every extrema in the test data.
        """
        self.setup()

        self.assertDictEqual(
            {
                1189: (0.117402, 429, 989),
                429: (0.256168, 410, 424),
                1170: (0.117402, 410, 970),
                410: (1.99362, 410, 410),
            },
            self.test_object.get_merge_sequence(),
            "The merge sequence does not match the expected output.",
        )

    def test_get_weights(self):
        """
        Function to test if default weights can be applied to the
        MorseComplex object. This feature should be evaluated
        in order to determine if it makes sense to keep it.
        """
        self.setup()

        equal_weights = np.ones(len(self.Y)) * 1.0 / float(len(self.Y))

        np.testing.assert_array_equal(
            equal_weights,
            self.test_object.get_weights(),
            "The default weights should be 1 for every row in the input.",
        )

        test_weights = self.test_object.get_weights(list(range(len(self.test_object.w))))

        np.testing.assert_array_equal(
            equal_weights,
            test_weights,
            "User should be able to filter the rows retrieved from get_weights.",
        )

        test_weights = self.test_object.get_weights([])

        np.testing.assert_array_equal(
            [], test_weights, "An empty query should return empty results."
        )

    def test_load_data_and_build(self):
        """
        Tests that loading the same test data from file yields an
        equivalent result
        """
        self.setup()

        test_object = topopy.MorseComplex()

        np.savetxt(
            "test_file.csv",
            np.hstack((self.X, np.atleast_2d(self.Y).T)),
            delimiter=",",
            header="x0,x1,y",
        )
        test_object.load_data_and_build("test_file.csv", delimiter=",")
        os.remove("test_file.csv")

        self.assertDictEqual(
            self.test_object.merge_sequence,
            test_object.merge_sequence,
            "loading from file should produce the same merge sequence",
        )
        self.assertDictEqual(
            dict(self.test_object.base_partitions),
            dict(test_object.base_partitions),
            "loading from file should produce the base partitions.",
        )

    def test_persistence(self):
        """
        Tests the getting and setting of different persistence
        values. Will also test that the number of components
        decreases correctly as persistence is increased
        """
        self.setup()
        self.test_object.set_persistence(self.test_object.persistences[1])
        self.assertEqual(
            self.test_object.persistences[1],
            self.test_object.get_persistence(),
            "Users should be able to get and set the persistence.",
        )

    def test_to_json(self):
        """
        Testing the output of the Morse Complex as a json object
        """
        self.setup()
        self.assertEqual(
            "{\"Hierarchy\":[{\"Persistence\":1.99362,\"Dying\":410,\"Surviving\":410,\"Saddle\":410},{\"Persistence\":0.256168,\"Dying\":429,\"Surviving\":410,\"Saddle\":424},{\"Persistence\":0.117402,\"Dying\":1170,\"Surviving\":410,\"Saddle\":970},{\"Persistence\":0.117402,\"Dying\":1189,\"Surviving\":429,\"Saddle\":989}],"
            "\"Partitions\":[410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,410,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,429,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1170,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189,1189]}",
            self.test_object.to_json(),
            "The hierarchy printed differs from what is expected.",
        )

    def test_reset(self):
        """
        Tests resetting of internal storage of the Morse Complex object
        """
        self.setup()
        self.test_object.reset()

        self.assertListEqual(
            [],
            self.test_object.persistences,
            "reset should clear all internal storage of the Morse Complex.",
        )
        self.assertDictEqual(
            {},
            self.test_object.base_partitions,
            "reset should clear all internal storage of the Morse Complex.",
        )
        self.assertEqual(
            0,
            self.test_object.persistence,
            "reset should clear all internal storage of the Morse Complex.",
        )
        self.assertDictEqual(
            {},
            self.test_object.merge_sequence,
            "reset should clear all internal storage of the Morse Complex.",
        )
        self.assertListEqual(
            [],
            self.test_object.max_indices,
            "reset should clear all internal storage of the Morse Complex.",
        )
        self.assertEqual(
            [],
            self.test_object.X,
            "reset should clear all internal storage of the Morse Complex."
        )
        self.assertEqual(
            [],
            self.test_object.Y,
            "reset should clear all internal storage of the Morse Complex."
        )
        self.assertEqual(
            [],
            self.test_object.w,
            "reset should clear all internal storage of the Morse Complex."
        )
        self.assertEqual(
            [],
            self.test_object.Xnorm,
            "reset should clear all internal storage of the Morse Complex.",
        )

    def test_save(self):
        """
        Testing the save feature correctly dumps to a json file.
        """
        self.setup()
        self.test_object.save("test.json")

        with open("mc_gold.json", "r") as data_file:
            gold_json = data_file.read()
            gold_json = json.loads(gold_json)

        with open("test.json", "r") as data_file:
            test_json = data_file.read()
            test_json = json.loads(test_json)

        self.assertDictEqual(
            gold_json,
            test_json,
            "save does not reproduce the same results as the gold custom json file.",
        )
        os.remove("test.json")

        self.test_object.save()
        with open("morse_complex.json", "r") as data_file:
            test_json = data_file.read()
            test_json = json.loads(test_json)

        self.assertDictEqual(
            gold_json,
            test_json,
            "save does not reproduce the same results as the gold default json file.",
        )
        os.remove("morse_complex.json")

    def test_shape_functions(self):
        """
        Test the get_dimensionality and get_sample_size functions
        """
        self.setup()

        self.assertEqual(
            self.X.shape[1],
            self.test_object.get_dimensionality(),
            "get_dimensionality should return the number of columns in X.",
        )
        self.assertEqual(
            self.X.shape[0],
            self.test_object.get_sample_size(),
            "get_sample_size should return the number of rows in X.",
        )
        self.assertEqual(
            240,
            self.test_object.get_sample_size(1189),
            "get_sample_size should return the number of rows in the specified partition.",
        )

    def test_get_partitions(self):
        """
        Test the ability to retrieve partition labels for individual
        query points and the entire dataset if not specified
        """
        self.setup()
        partitions = self.test_object.get_partitions()
        self.assertEqual(
            4,
            len(partitions),
            "The number of partitions without specifying the persistence should be the same as requesting the base (0) persistence level",
        )

        self.test_object = topopy.MorseComplex()
        partitions = self.test_object.get_partitions()
        self.assertEqual(
            {},
            partitions,
            "Requesting partitions on an unbuilt object should return an empty dictionary",
        )
