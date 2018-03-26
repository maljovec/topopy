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

import sys
import time

from .topology import MergeTreeFloat, vectorFloat, vectorString

from . import TopologicalObject


class MergeTree(TopologicalObject):
    """ A wrapper class for the C++ merge tree data structure
    """

    def __init__(self, graph='beta skeleton', gradient='steepest',
                 max_neighbors=-1, beta=1.0, normalization=None, connect=False,
                 aggregator=None, debug=False):
        """ Initialization method
            @ In, graph, an optional string specifying the type of
            neighborhood graph to use. Default is 'beta skeleton,' but
            other valid types are: 'delaunay,' 'relaxed beta skeleton,'
            'none', or 'approximate knn'
            @ In, gradient, an optional string specifying the type of
            gradient estimator to use. Currently the only available
            option is 'steepest'
            @ In, max_neighbors, an optional integer value specifying
            the maximum number of k-nearest neighbors used to begin a
            neighborhood search. In the case of graph='[relaxed] beta
            skeleton', we will begin with the specified approximate knn
            graph and prune edges that do not satisfy the empty region
            criteria.
            @ In, beta, an optional floating point value between 0 and
            2. This value is only used when graph='[relaxed] beta
            skeleton' and specifies the radius for the empty region
            graph computation (1=Gabriel graph, 2=Relative neighbor
            graph)
            @ In, normalization, an optional string specifying whether
            the inputs/output should be scaled before computing.
            Currently, two modes are supported 'zscore' and 'feature'.
            'zscore' will ensure the data has a mean of zero and a
            standard deviation of 1 by subtracting the mean and dividing
            by the variance. 'feature' scales the data into the unit
            hypercube.
            @ In, connect, an optional boolean flag for whether the
            algorithm should enforce the data to be a single connected
            component.
            @ In, aggregator, an optional string that specifies what
            type of aggregation to do when duplicates are found in the
            domain space. Default value is None meaning the code will
            error if duplicates are identified.
            @ In, debug, an optional boolean flag for whether debugging
            output should be enabled.
        """
        super(MergeTree, self).__init__(graph=graph, gradient=gradient,
                                        max_neighbors=max_neighbors, beta=beta,
                                        normalization=normalization,
                                        connect=connect, aggregator=aggregator,
                                        debug=debug)

    def build(self, X, Y, w=None, names=None, edges=None):
        """ Assigns data to this object and builds the Morse-Smale
            Complex
            @ In, X, an m-by-n array of values specifying m
            n-dimensional samples
            @ In, Y, a m vector of values specifying the output
            responses corresponding to the m samples specified by X
            @ In, w, an optional m vector of values specifying the
            weights associated to each of the m samples used. Default of
            None means all points will be equally weighted
            @ In, names, an optional list of strings that specify the
            names to associate to the n input dimensions and 1 output
            dimension. Default of None means input variables will be x0,
            x1, ..., x(n-1) and the output will be y
            @ In, edges, an optional list of custom edges to use as a
            starting point for pruning, or in place of a computed graph.
        """
        super(MergeTree, self).build(X, Y, w, names, edges)

        if self.debug:
            sys.stderr.write('Merge Tree Computation: ')
            start = time.clock()

        self.__tree = MergeTreeFloat(vectorFloat(self.Xnorm.flatten()),
                                     vectorFloat(self.Y),
                                     vectorString(self.names),
                                     str(self.gradient),
                                     self.graph_rep.full_graph(), self.debug)

        self.nodes = self.__tree.Nodes()
        self.edges = self.__tree.Edges()
        self.augmentedEdges = {}
        for key, val in self.__tree.AugmentedEdges().iteritems():
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

        if self.debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))

    def build_for_ContourTree(self, contour_tree):
        """
        """
        if self.debug:
            sys.stderr.write('Merge Tree Computation: ')
            start = time.clock()

        self.__tree = MergeTreeFloat(vectorFloat(contour_tree.flatten()),
                                     vectorFloat(contour_tree.Y),
                                     vectorString(contour_tree.names),
                                     str(contour_tree.gradient),
                                     contour_tree.graph_rep.full_graph(),
                                     self.debug)

        self.nodes = self.__tree.Nodes()
        self.edges = self.__tree.Edges()
        self.augmentedEdges = {}
        for key, val in self.__tree.AugmentedEdges().iteritems():
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

        if self.debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))
