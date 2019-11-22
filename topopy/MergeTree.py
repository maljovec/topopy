import sys
import time

from .topology import MergeTreeFloat, vectorFloat

from .TopologicalObject import TopologicalObject


class MergeTree(TopologicalObject):
    """A wrapper class for the C++ merge tree data structure.


    Parameters
    ----------
    graph : nglpy.Graph
        A graph object used for determining neighborhoods in gradient estimation
    gradient : str
        An optional string specifying the type of gradient estimator to use.
        Currently the only available option is 'steepest'.
    normalization : str
        An optional string specifying whether the inputs/output should be
        scaled before computing. Currently, two modes are supported 'zscore'
        and 'feature'. 'zscore' will ensure the data has a mean of zero and a
        standard deviation of 1 by subtracting the mean and dividing by the
        variance. 'feature' scales the data into the unit hypercube.
    aggregator : str
        An optional string that specifies what type of aggregation to do when
        duplicates are found in the domain space. Default value is None meaning
        the code will error if duplicates are identified.
    debug : bool
        An optional boolean flag for whether debugging output should be enabled.

    """

    def __init__(
        self,
        graph=None,
        gradient="steepest",
        normalization=None,
        aggregator=None,
        debug=False,
    ):
        """ Empties all internal storage containers


        Returns
        -------
        None

        """
        super(MergeTree, self).__init__(
            graph=graph,
            gradient=gradient,
            normalization=normalization,
            aggregator=aggregator,
            debug=debug,
        )

    def _internal_build(self):
        """ This function assumes the self.__tree object has been setup,
            though it doesn't care how. It will then setup all python
            side data structures to be used for querying this object.
        """
        self.nodes = self.__tree.Nodes()
        self.edges = self.__tree.Edges()
        self.augmentedEdges = {}
        for key, val in self.__tree.AugmentedEdges().items():
            self.augmentedEdges[key] = list(val)
        self.root = self.__tree.Root()

        seen = set()
        self.branches = set()

        # Find all of the branching nodes in the tree, degree > 1
        # That is, they appear in more than one edge
        for e1, e2 in self.edges:
            if e1 not in seen:
                seen.add(e1)
            else:
                self.branches.add(e1)

            if e2 not in seen:
                seen.add(e2)
            else:
                self.branches.add(e2)

        # The nodes that are not branches are leaves
        self.leaves = set(self.nodes.keys()) - self.branches
        self.leaves.remove(self.root)

    def build(self, X, Y, w=None):
        """Assigns data to this object and builds the Merge Tree.

        Uses an internal graph given in the constructor to build a merge tree
        on the passed in data. Weights are currently ignored.

        Parameters
        ----------
        X : np.ndarray
            An m-by-n array of values specifying m n-dimensional samples
        Y : np.array
            An m vector of values specifying the output responses corresponding
            to the m samples specified by X
        w : np.array
            An optional m vector of values specifying the weights associated to
            each of the m samples used. Default of None means all points will be
            equally weighted

        Returns
        -------
        None

        """
        super(MergeTree, self).build(X, Y, w)

        if self.debug:
            sys.stdout.write("Merge Tree Computation: ")
            start = time.perf_counter()

        self.__tree = MergeTreeFloat(
            vectorFloat(self.Xnorm.flatten()),
            vectorFloat(self.Y),
            str(self.gradient),
            self.graph.full_graph(),
            self.debug,
        )

        self._internal_build()

        if self.debug:
            end = time.perf_counter()
            sys.stdout.write("%f s\n" % (end - start))

    def _build_for_contour_tree(self, contour_tree, negate=False):
        """ A helper function that will reduce duplication of data by
            reusing the parent contour tree's parameters and data
        """
        if self.debug:
            tree_type = "Join"
            if negate:
                tree_type = "Split"
            sys.stdout.write("{} Tree Computation: ".format(tree_type))
            start = time.perf_counter()

        Y = contour_tree.Y
        if negate:
            Y = -Y

        self.__tree = MergeTreeFloat(
            vectorFloat(contour_tree.Xnorm.flatten()),
            vectorFloat(Y),
            str(contour_tree.gradient),
            contour_tree.graph.full_graph(),
            self.debug,
        )
        self._internal_build()
        if self.debug:
            end = time.perf_counter()
            sys.stdout.write("%f s\n" % (end - start))
