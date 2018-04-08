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
import numpy as np
import time
import operator
import warnings

import networkx as nx

from . import MergeTree
from . import TopologicalObject


class ContourTree(TopologicalObject):
    """ A class for computing a contour tree from two merge trees
    """
    def __init__(self, graph='beta skeleton', gradient='steepest',
                 max_neighbors=-1, beta=1.0, normalization=None, connect=False,
                 aggregator=None, debug=False, short_circuit=True):
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
            @ In, short_circuit, an optional boolean flag for whether the
            contour tree should be short circuited (TODO: fix description).
        """
        super(ContourTree, self).__init__(graph=graph, gradient=gradient,
                                          max_neighbors=max_neighbors,
                                          beta=beta,
                                          normalization=normalization,
                                          connect=connect,
                                          aggregator=aggregator,
                                          debug=debug)
        self.short_circuit = short_circuit

    def reset(self):
        """
            Empties all internal storage containers
        """
        super(ContourTree, self).reset()
        self.edges = []
        self.augmentedEdges = {}
        self.sortedNodes = []
        self.branches = set()
        self.superNodes = []
        self.superArcs = []

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
        super(ContourTree, self).build(X, Y, w, names, edges)

        # Build the join and split trees that we will merge into the
        # contour tree
        joinTree = MergeTree()
        splitTree = MergeTree()

        joinTree.build_for_ContourTree(self, True)
        splitTree.build_for_ContourTree(self, False)

        self.augmentedEdges = dict(joinTree.augmentedEdges)
        self.augmentedEdges.update(dict(splitTree.augmentedEdges))

        if self.short_circuit:
            jt = self._construct_nx_tree(joinTree, splitTree)
            st = self._construct_nx_tree(splitTree, joinTree)
        else:
            jt = self._construct_nx_tree(joinTree)
            st = self._construct_nx_tree(splitTree)

        self._process_tree(jt, st)
        self._process_tree(st, jt)

        # Now we have a fully augmented contour tree stored in nodes and
        # edges The rest is some convenience stuff for querying later

        self._identifyBranches()
        self._identifySuperGraph()

        if self.debug:
            sys.stderr.write('Sorting Nodes: ')
            start = time.clock()

        self.sortedNodes = sorted(enumerate(self.Y),
                                  key=operator.itemgetter(1))

        if self.debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))

    def _identifyBranches(self):
        """ A helper function for determining all of the branches in the
            tree. This should be called after the tree has been fully
            constructed and its nodes and edges are populated.
        """

        if self.debug:
            sys.stderr.write('Identifying branches: ')
            start = time.clock()

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

        if self.debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))

    def _identifySuperGraph(self):
        """ A helper function for determining the condensed
            representation of the tree. That is, one that does not hold
            all of the internal nodes of the graph. The results will be
            stored in ContourTree.superNodes and ContourTree.superArcs.
            These two can be used to potentially speed up queries by
            limiting the searching on the graph to only nodes on these
            super arcs.
        """

        if self.debug:
            sys.stderr.write('Condensing Graph: ')
            start = time.clock()

        G = nx.DiGraph()
        G.add_edges_from(self.edges)

        if self.short_circuit:
            self.superNodes = G.nodes()
            self.superArcs = G.edges()
            # There should be a way to populate this from the data we
            # have...
            return

        self.augmentedEdges = {}
        N = len(self.Y)
        processed = np.zeros(N)
        for node in range(N):

            # We can short circuit this here, since some of the nodes
            # will be handled within the while loops below.
            if processed[node]:
                continue

            # Loop through each internal node (see if below for
            # determining what is internal), trace up and down to a
            # node's first non-internal node in either direction
            # removing all of the internal nodes and pushing them into a
            # list. This list (removedNodes) will be put into a
            # dictionary keyed on the endpoints of the final super arc.
            if G.in_degree(node) == 1 and G.out_degree(node) == 1:
                # The sorted list of nodes that will be condensed by
                # this super arc
                removedNodes = []

                # Trace down to a non-internal node

                lowerLink = list(G.in_edges(node))[0][0]
                while (G.in_degree(lowerLink) == 1 and
                       G.out_degree(lowerLink) == 1):
                    newLowerLink = list(G.in_edges(lowerLink))[0][0]
                    G.add_edge(newLowerLink, node)
                    G.remove_node(lowerLink)
                    removedNodes.append(lowerLink)
                    lowerLink = newLowerLink

                removedNodes.reverse()
                removedNodes.append(node)

                # Trace up to a non-internal node
                upperLink = list(G.out_edges(node))[0][1]
                while (G.in_degree(upperLink) == 1 and
                       G.out_degree(upperLink) == 1):
                    newUpperLink = list(G.out_edges(upperLink))[0][1]
                    G.add_edge(node, newUpperLink)
                    G.remove_node(upperLink)
                    removedNodes.append(upperLink)
                    upperLink = newUpperLink

                G.add_edge(lowerLink, upperLink)
                G.remove_node(node)

                self.augmentedEdges[(lowerLink, upperLink)] = removedNodes

                # This is to help speed up the process by skipping nodes
                # we have already condensed, and to prevent us from not
                # being able to find nodes that have already been
                # removed.
                processed[removedNodes] = 1

        self.superNodes = G.nodes()
        self.superArcs = G.edges()

        if self.debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))

    def get_seeds(self, threshold, getPath=False):
        """ Returns a list of seed points for isosurface extraction
            given a threshold value
            @ In, threshold, float, the isovalue for which we want to
                identify seed points for isosurface extraction
        """
        seeds = []
        # ####### DEBUG Stuff ########
        paths = []
        # ##### END DEBUG Stuff ######
        for e1, e2 in self.superArcs:
            # Because we did some extra work in _process_tree, we can
            # safely assume e1 is lower than e2
            if self.Y[e1] <= threshold <= self.Y[e2]:
                # ####### DEBUG Stuff ########
                print('Super Arc: %d -> %d' % (e1, e2))
                # ##### END DEBUG Stuff ######
                if (e1, e2) in self.augmentedEdges:
                    path = []

                    # These should be sorted
                    edgeList = self.augmentedEdges[(e1, e2)]
                elif (e2, e1) in self.augmentedEdges:
                    # ####### DEBUG Stuff ########
                    print('\tSwapping arc order')
                    # ##### END DEBUG Stuff ######

                    e1, e2 = e2, e1

                    # These should be reverse sorted
                    edgeList = list(reversed(self.augmentedEdges[(e1, e2)]))
                else:
                    continue

                startNode = e1
                for endNode in edgeList + [e2]:
                    # ####### DEBUG Stuff ########
                    path.append((startNode, endNode))
                    # ##### END DEBUG Stuff ######
                    if self.Y[endNode] >= threshold:
                        # Stop when you find the first point above the
                        # threshold
                        break
                    startNode = endNode

                seeds.append(startNode)
                seeds.append(endNode)
                print('\t\t%d: %f' % (startNode, self.Y[startNode]))
                print('\t\t%d: %f' % (endNode, self.Y[endNode]))
                paths.append(path)
        if getPath:
            return seeds, paths
        else:
            return seeds

    def _construct_nx_tree(self, thisTree, thatTree=None):
        """ A function for creating networkx instances that can be used
            more efficiently for graph manipulation than the MergeTree
            class.
            @ In, thisTree, a MergeTree instance for which we will
                construct a networkx graph
            @ In, thatTree, a MergeTree instance optionally used to
                speed up the processing by bypassing the fully augmented
                search and only focusing on the partially augmented
                split and join trees
            @ Out, nxTree, a networkx.Graph instance matching the
                details of the input tree.
        """
        if self.debug:
            sys.stderr.write('Networkx Tree construction: ')
            start = time.clock()

        nxTree = nx.DiGraph()
        nxTree.add_edges_from(thisTree.edges)

        nodesOfThatTree = []
        if thatTree is not None:
            nodesOfThatTree = thatTree.nodes.keys()

        # Fully or partially augment the join tree
        for (superNode, _), nodes in thisTree.augmentedEdges.items():
            superNodeEdge = list(nxTree.out_edges(superNode))
            if len(superNodeEdge) > 1:
                warnings.warn('The supernode {} should have only a single '
                              'emanating edge. Merge tree is invalidly '
                              'structured'.format(superNode))
            endNode = superNodeEdge[0][1]
            startNode = superNode
            nxTree.remove_edge(startNode, endNode)
            for node in nodes:
                if thatTree is None or node in nodesOfThatTree:
                    nxTree.add_edge(startNode, node)
                    startNode = node

            # Make sure this is not the root node trying to connect to
            # itself
            if startNode != endNode:
                nxTree.add_edge(startNode, endNode)

        if self.debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))

        return nxTree

    def _process_tree(self, thisTree, thatTree):
        """ A function that will process either a split or join tree
            with reference to the other tree and store it as part of
            this CT instance.
            @ In, thisTree, a networkx.Graph instance representing a
                merge tree for which we will process all of its leaf
                nodes into this CT object
            @ In, thatTree, a networkx.Graph instance representing the
                opposing merge tree which will need to be updated as
                nodes from thisTree are processed
            @ Out, None
        """
        if self.debug:
            sys.stderr.write('Processing Tree: ')
            start = time.clock()

        # Get all of the leaf nodes that are not branches in the other
        # tree
        if len(thisTree.nodes()) > 1:
            leaves = set([v for v in thisTree.nodes()
                          if thisTree.in_degree(v) == 0 and
                          thatTree.in_degree(v) < 2])
        else:
            leaves = set()

        while len(leaves) > 0:
            v = leaves.pop()

            # if self.debug:
            #     sys.stderr.write('\tProcessing {} -> {}\n'
            #                      .format(v, thisTree.edges(v)[0][1]))

            # Take the leaf and edge out of the input tree and place it
            # on the CT
            edges = list(thisTree.out_edges(v))
            if len(edges) != 1:
                warnings.warn('The node {} should have a single emanating '
                              'edge.\n'.format(v))
            e1 = edges[0][0]
            e2 = edges[0][1]
            # This may be a bit beside the point, but if we want all of
            # our edges pointing 'up,' we can verify that the edges we
            # add have the lower vertex pointing to the upper vertex.
            # This is useful only for nicely plotting with some graph
            # tools (graphviz/networkx), and I guess for consistency
            # sake.
            if self.Y[e1] < self.Y[e2]:
                self.edges.append((e1, e2))
            else:
                self.edges.append((e2, e1))

            # Removing the node will remove its constituent edges from
            # thisTree
            thisTree.remove_node(v)

            # This is the root node of the other tree
            if thatTree.out_degree(v) == 0:
                thatTree.remove_node(v)
                # if self.debug:
                #     sys.stderr.write('\t\tRemoving root {} from other tree\n'
                #                      .format(v))
            # This is a "regular" node in the other tree, suppress it
            # there, but be sure to glue the upper and lower portions
            # together
            else:
                # The other ends of the node being removed are added to
                # "that" tree

                if len(thatTree.in_edges(v)) > 0:
                    startNode = list(thatTree.in_edges(v))[0][0]
                else:
                    # This means we are at the root of the other tree,
                    # we can safely remove this node without connecting
                    # its predecessor with its descendant
                    startNode = None

                if len(thatTree.out_edges(v)) > 0:
                    endNode = list(thatTree.out_edges(v))[0][1]
                else:
                    # This means we are at a leaf of the other tree,
                    # we can safely remove this node without connecting
                    # its predecessor with its descendant
                    endNode = None

                if startNode is not None and endNode is not None:
                    thatTree.add_edge(startNode, endNode)

                thatTree.remove_node(v)

                # if self.debug:
                #     sys.stderr.write('\t\tSuppressing {} in other tree and '
                #                      'gluing {} to {}\n'
                #                      .format(v, startNode, endNode))

            if len(thisTree.nodes()) > 1:
                leaves = set([v for v in thisTree.nodes()
                              if thisTree.in_degree(v) == 0 and
                              thatTree.in_degree(v) < 2])
            else:
                leaves = set()

            # if self.debug:
            #     myMessage = '\t\tValid leaves: '
            #     sep = ''
            #     for leaf in leaves:
            #         myMessage += sep + str(leaf)
            #         sep = ','
            #     sys.stderr.write(myMessage+'\n')

        if self.debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))
