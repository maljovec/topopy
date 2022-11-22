"""
    This module will test the basic functionalities of
    topopy.TopologicalObject
"""
import os
import sys
import warnings
from unittest import TestCase

import nglpy as ngl
import numpy as np
import sklearn

import topopy

from .test_functions import generate_test_grid_2d, gerber


class TestTO(TestCase):
    """
    Class for testing the base class Topological Object
    """

    def setup(self):
        """
        Setup function will create a fixed point set and parameter
        settings for testing different aspects of this library.
        """
        self.X = generate_test_grid_2d(10)
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
        self.to = topopy.TopologicalObject(debug=False, graph=self.graph)
        self.to.build(self.X, self.Y)

    def test_aggregation(self):
        # Since the default function can change, here we will only test
        # that the correct number of each array is reported
        X = np.ones((11, 2))
        X[10] = [0, 0]
        Y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

        warnings.filterwarnings("ignore")

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y)
        self.assertEqual(
            2,
            len(x),
            "aggregate_duplicates should return a list of " "unique items in X.",
        )
        self.assertEqual(
            2,
            len(y),
            "aggregate_duplicates should return the aggregated "
            "values of Y for each unique item in X.",
        )

        # Next, we will test each of the string aggregation function
        # names on scalar Y values
        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "min")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [100, 0])

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "max")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [100, 9])

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "mean")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [100, 4.5])

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "average")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [100, 4.5])

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "median")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [100, 4.5])

        # Next, we will test each of the string aggregation function
        # names on vector Y values
        Y = np.array(
            [
                [0, 9],
                [1, 8],
                [2, 7],
                [3, 6],
                [4, 5],
                [5, 4],
                [6, 3],
                [7, 2],
                [8, 1],
                [9, 0],
                [100, 0],
            ]
        )

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "min")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [[100, 0], [0, 0]])

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "max")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [[100, 0], [9, 9]])

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "mean")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [[100, 0], [4.5, 4.5]])

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "median")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [[100, 0], [4.5, 4.5]])

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "first")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [[100, 0], [0, 9]])

        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "last")
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [[100, 0], [9, 0]])

        # Testing custom callable aggregator
        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, lambda x: x[0])
        self.assertListEqual(x.tolist(), [[0, 0], [1, 1]])
        self.assertListEqual(y.tolist(), [[100, 0], [0, 9]])

        warnings.filterwarnings("always")
        # Testing an invalid aggregator
        with warnings.catch_warnings(record=True) as w:
            x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y, "invalid")

            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertEqual(
                'Aggregator "invalid" not understood. Skipping sample ' "aggregation.",
                str(w[-1].message),
            )

            self.assertListEqual(x.tolist(), X.tolist())
            self.assertListEqual(y.tolist(), Y.tolist())

        # Testing aggregator on non-duplicate data
        X = np.array([[0, 0], [0, 1]])
        Y = np.array([0, 1])
        x, y = topopy.TopologicalObject.aggregate_duplicates(X, Y)
        self.assertListEqual(x.tolist(), X.tolist())
        self.assertListEqual(y.tolist(), Y.tolist())

        warnings.filterwarnings("ignore")
        # Testing use of the aggregator in the check_duplicates function
        X = np.ones((11, 2))
        X[10] = [0, 0]
        Y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

        to = topopy.TopologicalObject()
        self.assertRaises(ValueError, to.build, **{"X": X, "Y": Y})
        graph = ngl.EmptyRegionGraph(max_neighbors=10)
        to = topopy.TopologicalObject(aggregator="mean", graph=graph)
        to.build(X, Y)
        warnings.filterwarnings("always")

    def test_empty(self):
        """
        Test the ability to handle None objects as input
        """
        self.setup()
        self.to = topopy.TopologicalObject()
        self.to.build(None, None)

    def test_debug(self):
        """
        Test the debugging output of the TopologicalObject
        """
        self.setup()
        test_file = "to_test_debug.txt"
        sys.stdout = open(test_file, "w")

        self.to = topopy.TopologicalObject(debug=True, graph=self.graph)
        self.to.build(self.X, self.Y)
        sys.stdout.close()

        lines = ["Graph Preparation:"]

        with open(test_file, "r") as fp:
            debug_output = fp.read()
            for line in lines:
                self.assertIn(line, debug_output)

        os.remove(test_file)
        # Restore stdout
        sys.stdout = sys.__stdout__

    def test_get_normed_x(self):
        """
        Tests get_normed_x in several different contexts:
            Single Element extraction
            Single Column extraction
            Single Row extraction
            Multiple row extraction
            Multiple column extraction
            Full data extraction
        """
        self.setup()

        for norm, X in self.norm_x.items():
            to = topopy.TopologicalObject(normalization=norm, graph=self.graph)
            to.build(self.X, self.Y)

            # Test single column extraction
            for col in range(X.shape[1]):
                column_values = to.get_normed_x(cols=col)
                np.testing.assert_array_equal(
                    X[:, col],
                    column_values,
                    "get_normed_x should be able "
                    + "to access a full column of "
                    + "the input data.",
                )

            # Test single row extraction
            for row in range(X.shape[0]):
                row_values = to.get_normed_x(row).flatten()
                np.testing.assert_array_equal(
                    X[row, :],
                    row_values,
                    "get_normed_x should be able "
                    + "to access a full row of the "
                    + "input data.",
                )
                # Test single element extraction
                for col in range(X.shape[1]):
                    self.assertEqual(
                        X[row, col],
                        to.get_normed_x(row, col),
                        "get_normed_x should be able to access "
                        + "a single element of the input data.",
                    )

            # Multiple row extraction
            row_values = to.get_normed_x(list(range(X.shape[0])), 0)
            np.testing.assert_array_equal(
                X[:, 0],
                row_values,
                "get_normed_x should be able to "
                + "access multiple rows of the "
                + "input data.",
            )

            # Multiple column extraction
            col_values = to.get_normed_x(0, list(range(X.shape[1]))).flatten()
            np.testing.assert_array_equal(
                X[0, :],
                col_values,
                "get_normed_x should be able to "
                + "access multiple columns of the "
                + "input data.",
            )

            # Full data extraction
            np.testing.assert_array_equal(
                X,
                to.get_normed_x(),
                "get_normed_x should be able to " + "access the entire input data.",
            )

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
            column_values = self.to.get_x(cols=col)
            np.testing.assert_array_equal(
                self.X[:, col],
                column_values,
                "get_x should be able to access a " + "full column of the input data.",
            )

        # Test single row extraction
        for row in range(self.X.shape[0]):
            row_values = self.to.get_x(row).flatten()
            np.testing.assert_array_equal(
                self.X[row, :],
                row_values,
                "get_x should be able to access a " + "full row of the input data.",
            )
            # Test single element extraction
            for col in range(self.X.shape[1]):
                self.assertEqual(
                    self.X[row, col],
                    self.to.get_x(row, col),
                    "get_x should be able to access a single "
                    + "element of the input data.",
                )

        # Multiple row extraction
        row_values = self.to.get_x(list(range(self.X.shape[0])), 0)
        np.testing.assert_array_equal(
            self.X[:, 0],
            row_values,
            "get_x should be able to access " + "multiple rows of the input data.",
        )

        # Multiple column extraction
        col_values = self.to.get_x(0, list(range(self.X.shape[1]))).flatten()
        np.testing.assert_array_equal(
            self.X[0, :],
            col_values,
            "get_x should be able to access " + "multiple columns of the input data.",
        )

        # Full data extraction
        np.testing.assert_array_equal(
            self.X,
            self.to.get_x(),
            "get_x should be able to access the entire input data.",
        )

        # Empty query
        np.testing.assert_array_equal(
            [],
            self.to.get_x([]),
            "get_x should be able to access " + "return an empty array on null filter.",
        )

    def test_get_y(self):
        """
        Tests get_y in several different contexts:
            Single Element extraction
            Multiple row extraction
            Full data extraction
        """
        self.setup()

        for row in range(len(self.Y)):
            # Test single element extraction
            self.assertEqual(
                self.Y[row],
                self.to.get_y(row),
                "get_y should be able to access a single "
                + "element of the input data.",
            )

        # Multiple row extraction
        row_values = self.to.get_y(list(range(len(self.Y))))
        np.testing.assert_array_equal(
            self.Y,
            row_values,
            "get_y should be able to access " + "multiple rows of the input data.",
        )

        # Full data extraction
        np.testing.assert_array_equal(
            self.Y,
            self.to.get_y(),
            "get_y should be able to access the entire input data.",
        )

        # Empty query
        np.testing.assert_array_equal(
            [],
            self.to.get_y([]),
            "get_y should be able to access " + "return an empty array on null filter.",
        )

    def test_neighbors(self):
        """
        Tests the ability to retrieve the neighbors of a given index
        """
        self.setup()
        self.assertSetEqual(
            {10, 1},
            set(self.to.get_neighbors(0)),
            "get_neighbors should return a list of integer "
            + "indices indicating who is connected to the "
            + "given index.",
        )

    def test_reset(self):
        """
        Tests resetting of internal storage of the to object
        """
        self.setup()
        self.to.reset()

        self.assertEqual(
            [],
            self.to.X,
            "reset should clear all internal storage of the to.",
        )
        self.assertEqual(
            [],
            self.to.Y,
            "reset should clear all internal storage of the to.",
        )
        self.assertEqual(
            [],
            self.to.Xnorm,
            "reset should clear all internal storage of the to.",
        )

    def test_shape_functions(self):
        """
        Test the get_dimensionality and get_sample_size functions
        """
        self.setup()

        self.assertEqual(
            self.X.shape[1],
            self.to.get_dimensionality(),
            "get_dimensionality should return the number of columns in X.",
        )
        self.assertEqual(
            self.X.shape[0],
            self.to.get_sample_size(),
            "get_sample_size should return the number of rows in X.",
        )
