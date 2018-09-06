"""
    This module will test the basic functionalities of
    topopy.MergeTree
"""
from unittest import TestCase
import numpy as np
import topopy
from .test_functions import gerber, generate_test_grid_2d
import sklearn


class TestMT(TestCase):
    """
    Class for testing the Contour Tree and its prerequisite the Merge Tree
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

    def test_debug(self):
        """
        Testing if we can build the Merge Tree directly
        """
        self.setup()

        mt = topopy.MergeTree(debug=True, max_neighbors=10)
        mt.build(self.X, self.Y)

    def test_merge_tree(self):
        """
        Testing if we can build the Merge Tree directly
        """
        self.setup()

        mt = topopy.MergeTree(debug=False, max_neighbors=10)
        mt.build(self.X, self.Y)

        self.assertEqual(
            9,
            len(mt.leaves),
            "The 2D Gerber test function " "should have 9 leaves in its split tree",
        )
        self.assertEqual(
            8,
            len(mt.branches),
            "The 2D Gerber test function " "should have 8 branches in its split tree",
        )

        mt.build(self.X, -self.Y)

        self.assertEqual(
            4,
            len(mt.leaves),
            "The 2D Gerber test function " "should have 4 leaves in its join tree",
        )
        self.assertEqual(
            3,
            len(mt.branches),
            "The 2D Gerber test function " "should have 3 branches in its join tree",
        )
