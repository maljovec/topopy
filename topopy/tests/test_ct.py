"""
    This module will test the basic functionalities of
    topopy.ContourTree and by proxy topopy.MergeTree
"""
import os
import sys
from unittest import TestCase

import nglpy as ngl
import numpy as np
import sklearn

import topopy

from .test_functions import generate_test_grid_2d, gerber


class TestCT(TestCase):
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

        # Methods covered here:
        # __init__
        # build
        # __set_data
        self.graph = ngl.EmptyRegionGraph(max_neighbors=10)
        self.ct = topopy.ContourTree(graph=self.graph, debug=False)
        self.ct.build(self.X, self.Y)

    def test_debug(self):
        """
        Test the debugging output of the CT
        """
        self.setup()
        test_file = "ct_test_debug.txt"
        sys.stdout = open(test_file, "w")

        self.ct = topopy.ContourTree(debug=True, short_circuit=True, graph=self.graph)
        self.ct.build(self.X, self.Y)

        self.ct = topopy.ContourTree(debug=True, short_circuit=False, graph=self.graph)
        self.ct.build(self.X, self.Y)

        sys.stdout.close()

        lines = [
            "Graph Preparation:",
            "Split Tree Computation:",
            "Join Tree Computation:",
            "Networkx Tree construction:",
            "Networkx Tree construction:",
            "Processing Tree:",
            "Processing Tree:",
            "Identifying branches:",
            "Condensing Graph: ",
            "Sorting Nodes:",
            "Graph Preparation:",
            "Split Tree Computation:",
            "Join Tree Computation:",
            "Networkx Tree construction:",
            "Networkx Tree construction:",
            "Processing Tree:",
            "Processing Tree:",
            "Identifying branches:",
            "Condensing Graph:",
            "Sorting Nodes:",
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
        Test the build process of the ContourTree
        """
        self.setup()

        self.assertEqual(
            24,
            len(self.ct.superNodes),
            "The 2D Gerber test function should have 24 "
            + "nodes in its contour tree.",
        )
        self.assertEqual(
            23,
            len(self.ct.superArcs),
            "The 2D Gerber test function should have 23 " + "arcs in its contour tree.",
        )

    def test_no_short_circuit(self):
        """
        Test the build process of the ContourTree
        """
        self.setup()
        self.ct = topopy.ContourTree(graph=self.graph, short_circuit=False)
        self.ct.build(self.X, self.Y)

        self.assertEqual(
            71,
            len(self.ct.superNodes),
            "The 2D Gerber test function should have 71 "
            + "nodes in its contour tree.",
        )
        self.assertEqual(
            70,
            len(self.ct.superArcs),
            "The 2D Gerber test function should have 70 " + "arcs in its contour tree.",
        )

    def test_get_seeds(self):
        """
        Test the build process of the ContourTree
        """
        self.setup()
        seeds = self.ct.get_seeds(np.min(self.Y))
        self.assertEqual(
            2,
            len(seeds),
            "The 2D Gerber test function should "
            "have 2 seed points at the minimum levelset",
        )

        seeds = self.ct.get_seeds(0.5)
        self.assertEqual(
            4,
            len(seeds),
            "The 2D Gerber test function should "
            "have 4 seed points at the 0.5 levelset",
        )

        seeds = self.ct.get_seeds(np.max(self.Y))
        self.assertEqual(
            2,
            len(seeds),
            "The 2D Gerber test function should "
            "have 2 seed points at the maximum levelset",
        )

    # def test_persistence(self):
    #     """
    #     Tests the getting and setting of different persistence
    #     values. Will also test that the number of components
    #     decreases correctly as persistence is increased
    #     """
    #     self.setup()
    #     self.ct.set_persistence(self.ct.persistences[1])
    #     self.assertEqual(self.ct.persistences[1], self.ct.get_persistence(),
    #                      'Users should be able to get and set the ' +
    #                      'persistence.')
