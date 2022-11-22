"""
    This module will test the basic functionalities of
    topopy.MorseSmaleComplex and by proxy the topopy.MorseComplex
"""
import json
import os
import sys
from unittest import TestCase

import nglpy as ngl
import numpy as np
import sklearn

import topopy

from .test_functions import generate_test_grid_2d, gerber


class TestMSC(TestCase):
    """
    Class for testing the Morse-Smale Complex
    """

    def __init__(self, *args, **kwargs):
        super(TestMSC, self).__init__(*args, **kwargs)

        # Python 2.7 compatibility
        if not hasattr(self, "assertCountEqual"):
            self.assertCountEqual = self.assertItemsEqual

    def setup(self):
        """
        Setup function will create a fixed point set and parameter
        settings for testing different aspects of this library.
        """
        self.X = generate_test_grid_2d(40)
        self.Y = gerber(self.X)
        self.graph = ngl.EmptyRegionGraph(max_neighbors=10)

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
        self.test_object = topopy.MorseSmaleComplex(debug=False, graph=self.graph)
        self.test_object.build(self.X, self.Y)

        gold_path = os.path.join("topopy", "tests", "msc_gold.json")
        with open(gold_path, "r") as data_file:
            gold_json = data_file.read()
            gold_json = json.loads(gold_json)
        self.gold = gold_json

    def test_debug(self):
        """
        Test the debugging output of the MSC
        """
        self.setup()

        test_file = "msc_test_debug.txt"
        sys.stdout = open(test_file, "w")

        self.test_object = topopy.MorseSmaleComplex(debug=True, graph=self.graph)
        self.test_object.build(self.X, self.Y)

        sys.stdout.close()
        lines = [
            "Graph Preparation:",
            "Decomposition:",
            "Stable Decomposition:",
            "Unstable Decomposition:",
        ]

        with open(test_file, "r") as fp:
            debug_output = fp.read()
            for line in lines:
                self.assertIn(line, debug_output)

        os.remove(test_file)
        # Restore stdout
        sys.stdout = sys.__stdout__

    def test_default(self):
        """
        Test the build process of the MorseSmaleComplex
        """
        self.setup()

        self.assertEqual(
            len(self.test_object.base_partitions.keys()),
            16,
            "The 2D Gerber test function should have 16 "
            "partitions at the base level.",
        )

    def test_get_classification(self):
        """
        Testing the ability to retrieve the correct classification
        for a minimum, maximum, and regular data point
        """
        self.setup()

        self.assertEqual(
            "minimum",
            self.test_object.get_classification(0),
            "get_classification has misidentified a local minimum",
        )
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
                (1599, 1189),
                (1584, 1170),
                (984, 410),
                (999, 429),
                (999, 1189),
                (1584, 1189),
                (0, 410),
                (960, 410),
                (24, 410),
                (960, 1170),
                (24, 429),
                (984, 1170),
                (984, 429),
                (1560, 1170),
                (984, 1189),
                (39, 429),
            },
            set(self.test_object.get_current_labels()),
            "The base "
            "partition labels returned from " + "get_current_labels does not match.",
        )

        self.test_object.set_persistence(self.test_object.persistences[-1])
        self.assertSetEqual(
            {(1599, 410)},
            set(self.test_object.get_current_labels()),
            "The base "
            "partition labels returned from " + "get_current_labels does not match.",
        )

    def test_get_label(self):
        """
        Testing the ability to retrieve the min-max label of any
        point in the data.
        """
        self.setup()

        self.assertEqual(
            (0, 410),
            self.test_object.get_label(0),
            "The label of a single index should be retrievable.",
        )

        self.assertEqual(
            len(self.Y),
            len(self.test_object.get_label()),
            "The label of the entire dataset should be retrievable.",
        )

        gold_labels = np.array([[0, 410], [0, 410]])
        test_labels = np.array(self.test_object.get_label([0, 1]).flatten().tolist())
        np.testing.assert_array_equal(
            gold_labels,
            test_labels,
            "The label of a multiple indices should be retrievable.",
        )

        self.assertEqual(
            len(self.Y),
            len(self.test_object.get_label()),
            "Requesting "
            "labels without specifying an index should return a "
            "list of all labels",
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

        merge_sequence = {}
        for item in self.gold["Hierarchy"]:
            merge_sequence[item["Dying"]] = (
                item["Persistence"],
                item["Surviving"],
                item["Saddle"],
            )

        self.assertDictEqual(
            merge_sequence,
            self.test_object.get_merge_sequence(),
            "The merge sequence does not match the " + "expected output.",
        )

    def test_get_weights(self):
        """
        Function to test if default weights can be applied to the
        MorseSmaleComplex object. This feature should be evaluated
        in order to determine if it makes sense to keep it.
        """
        self.setup()

        equal_weights = np.ones(len(self.Y)) * 1.0 / float(len(self.Y))

        np.testing.assert_array_equal(
            equal_weights,
            self.test_object.get_weights(),
            "The default weights should be 1 for every row in the input.",
        )

        test_weights = self.test_object.get_weights(
            list(range(len(self.test_object.w)))
        )

        np.testing.assert_array_equal(
            equal_weights,
            test_weights,
            "Should be able to filter the rows retrieved from get_weights.",
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

        local_object = topopy.MorseSmaleComplex()

        np.savetxt(
            "test_file.csv",
            np.hstack((self.X, np.atleast_2d(self.Y).T)),
            delimiter=",",
            header="x0,x1,y",
        )
        local_object.load_data_and_build("test_file.csv", delimiter=",")
        os.remove("test_file.csv")

        self.assertDictEqual(
            self.test_object.merge_sequence,
            local_object.merge_sequence,
            "loading from file should produce the same hierarchy",
        )
        self.assertSetEqual(
            set(self.test_object.base_partitions.keys()),
            set(local_object.base_partitions.keys()),
            "loading from file should produce the base partitions.",
        )

        for key in self.test_object.base_partitions.keys():
            self.assertListEqual(
                self.test_object.base_partitions[key].tolist(),
                local_object.base_partitions[key].tolist(),
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
            "Users should be able to get and set the " + "persistence.",
        )

    def test_to_json(self):
        """
        Testing the output of the Morse-Smale Complex as a json object
        """
        self.setup()

        test_json = json.loads(self.test_object.to_json())

        self.assertCountEqual(
            self.gold["Hierarchy"],
            test_json["Hierarchy"],
            "The hierarchy printed differs from what is expected.",
        )

        self.assertListEqual(
            self.gold["Partitions"],
            test_json["Partitions"],
            "The base partitions do not match",
        )

    def test_reset(self):
        """
        Tests resetting of internal storage of the msc object
        """
        self.setup()
        self.test_object.reset()

        self.assertListEqual(
            [],
            self.test_object.persistences,
            "reset should clear all internal storage of the msc.",
        )
        self.assertDictEqual(
            {},
            self.test_object.base_partitions,
            "reset should clear all internal storage of the msc.",
        )
        self.assertEqual(
            0,
            self.test_object.persistence,
            "reset should clear all internal storage of the msc.",
        )
        self.assertDictEqual(
            {},
            self.test_object.merge_sequence,
            "reset should clear all internal storage of the msc.",
        )
        self.assertListEqual(
            [],
            self.test_object.min_indices,
            "reset should clear all internal storage of the msc.",
        )
        self.assertListEqual(
            [],
            self.test_object.max_indices,
            "reset should clear all internal storage of the msc.",
        )
        self.assertEqual(
            [],
            self.test_object.X,
            "reset should clear all internal storage of the msc.",
        )
        self.assertEqual(
            [],
            self.test_object.Y,
            "reset should clear all internal storage of the msc.",
        )
        self.assertEqual(
            [],
            self.test_object.w,
            "reset should clear all internal storage of the msc.",
        )
        self.assertEqual(
            [],
            self.test_object.Xnorm,
            "reset should clear all internal storage of the msc.",
        )

    def test_save(self):
        """
        Testing the save feature correctly dumps to a json file.
        """
        self.setup()
        self.test_object.save("test.json")

        with open("test.json", "r") as data_file:
            test_json = data_file.read()
            test_json = json.loads(test_json)

        self.assertCountEqual(
            self.gold["Hierarchy"],
            test_json["Hierarchy"],
            "The hierarchy differs from what is expected.",
        )

        self.assertListEqual(
            self.gold["Partitions"],
            test_json["Partitions"],
            "The base partitions do not match",
        )

        os.remove("test.json")

        self.test_object.save()
        with open("morse_smale_complex.json", "r") as data_file:
            test_json = data_file.read()
            test_json = json.loads(test_json)

        self.assertCountEqual(
            self.gold["Hierarchy"],
            test_json["Hierarchy"],
            "The hierarchy differs from what is expected.",
        )

        self.assertListEqual(
            self.gold["Partitions"],
            test_json["Partitions"],
            "The base partitions do not match.",
        )

        os.remove("morse_smale_complex.json")

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
            121,
            self.test_object.get_sample_size((1599, 1189)),
            "get_sample_size should return the number of "
            + "rows in the specified partition.",
        )

    def test_get_partitions(self):
        self.setup()

        gold_msc_partitions = {}
        gold_stable_partitions = {}
        gold_unstable_partitions = {}
        for i, (min_label, max_label) in enumerate(self.gold["Partitions"]):
            if max_label not in gold_stable_partitions:
                gold_stable_partitions[max_label] = []
            gold_stable_partitions[max_label].append(i)

            if min_label not in gold_unstable_partitions:
                gold_unstable_partitions[min_label] = []
            gold_unstable_partitions[min_label].append(i)

            if (min_label, max_label) not in gold_msc_partitions:
                gold_msc_partitions[(min_label, max_label)] = []
            gold_msc_partitions[(min_label, max_label)].append(i)

        partitions = self.test_object.get_partitions()
        self.assertEqual(
            16,
            len(partitions),
            "The number of partitions "
            "without specifying the "
            "persistence should be the "
            "same as requesting the base "
            "(0) persistence level",
        )

        self.assertDictEqual(
            gold_msc_partitions,
            partitions,
            "The base partitions of the Morse-Smale complex should match",
        )

        partitions = self.test_object.get_stable_manifolds()
        self.assertEqual(
            4,
            len(partitions),
            "The number of stable manifolds should be 4 at the base level",
        )
        self.assertDictEqual(
            gold_stable_partitions,
            partitions,
            "The base partitions of the stable manifolds should match",
        )

        partitions = self.test_object.get_unstable_manifolds()
        self.assertEqual(
            9,
            len(partitions),
            "The number of unstable manifolds should be 9 at the base level",
        )
        self.assertDictEqual(
            gold_unstable_partitions,
            partitions,
            "The base partitions of the unstable manifolds should match",
        )

        test_p = 0.5

        merge_pattern = {}
        # Initialize every extrema to point to itself
        for merge in self.gold["Hierarchy"]:
            merge_pattern[merge["Dying"]] = merge["Dying"]

        # Now point to the correct label for the specified persistence
        for merge in self.gold["Hierarchy"]:
            if merge["Persistence"] < test_p:
                if merge["Surviving"] not in merge_pattern:
                    merge_pattern[merge["Surviving"]] = merge["Surviving"]
                merge_pattern[merge["Dying"]] = merge_pattern[merge["Surviving"]]

        gold_msc_partitions = {}
        gold_stable_partitions = {}
        gold_unstable_partitions = {}
        for i, (min_label, max_label) in enumerate(self.gold["Partitions"]):
            min_label = merge_pattern[min_label]
            max_label = merge_pattern[max_label]

            if max_label not in gold_stable_partitions:
                gold_stable_partitions[max_label] = []
            gold_stable_partitions[max_label].append(i)

            if min_label not in gold_unstable_partitions:
                gold_unstable_partitions[min_label] = []
            gold_unstable_partitions[min_label].append(i)

            if (min_label, max_label) not in gold_msc_partitions:
                gold_msc_partitions[(min_label, max_label)] = []
            gold_msc_partitions[(min_label, max_label)].append(i)

        partitions = self.test_object.get_partitions(test_p)
        self.assertEqual(
            4,
            len(partitions),
            "The number of partitions at the 0.5 level should be 4",
        )
        self.assertDictEqual(
            gold_msc_partitions,
            partitions,
            "The partitions of the Morse-Samle complex should match at "
            "the test level p={}".format(test_p),
        )

        partitions = self.test_object.get_stable_manifolds(test_p)
        self.assertEqual(
            1,
            len(partitions),
            "The number of stable manifolds should be 1 at the test "
            "level p={}".format(test_p),
        )
        self.assertDictEqual(
            gold_stable_partitions,
            partitions,
            "The base partitions of the stable manifolds should match "
            "at the test level p={}".format(test_p),
        )

        partitions = self.test_object.get_unstable_manifolds(0.5)
        self.assertEqual(
            4,
            len(partitions),
            "The number of unstable manifolds should be 9 at the test "
            "level p={}".format(test_p),
        )
        self.assertDictEqual(
            gold_unstable_partitions,
            partitions,
            "The base partitions of the unstable manifolds should "
            "match at the test level p={}".format(test_p),
        )

        self.test_object = topopy.MorseSmaleComplex()
        partitions = self.test_object.get_partitions()
        self.assertEqual(
            {},
            partitions,
            "Requesting partitions on an unbuilt object should return " "an empty dict",
        )

    def test_unstructured(self):
        """
        Tests functionality on something other than a regular grid
        """
        self.X = np.array(
            [
                [0, 0],
                [1, 1],
                [1, 0],
                [0, 1],
                [0.25, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
                [0.75, 0.25],
                [0.0, 0.6],
                [0.0, 0.25],
                [0.0, 0.75],
                [0.25, 0.0],
                [0.6, 0.0],
                [0.6, 0.25],
                [0.625, 0.55],
                [0.6, 0.75],
                [0.75, 0.0],
                [0.25, 0.55],
                [1, 0.25],
                [0.65, 1],
                [1, 0.55],
                [1, 0.75],
                [0.75, 1],
                [0.25, 1],
                [0.75, 0.6],
            ]
        )
        self.Y = gerber(self.X)

        self.test_object = topopy.MorseSmaleComplex(
            gradient="steepest",
            normalization=None,
            simplification="difference",
            aggregator=None,
            debug=False,
        )
        self.test_object.build(self.X, self.Y)

        self.assertEqual(
            13,
            len(self.test_object.get_merge_sequence()),
            "The merge sequence does not match the expected output.",
        )
