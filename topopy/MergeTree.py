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

from .topology import MergeTreeFloat, vectorFloat, vectorString, vectorInt

import sklearn.neighbors
import sklearn.linear_model
import sklearn.preprocessing


class MergeTree(object):
    """ A wrapper class for the C++ merge tree data structure
    """

    def __init__(self, X, Y, names=None, graph='beta skeleton',
                 gradient='steepest', knn=-1, beta=1.0, normalization=None,
                 edges=None, debug=False):
        """ Initialization method that takes at minimum a set of input
            points and corresponding output responses.
            @ In, X, an m-by-n array of values specifying m
                n-dimensional samples
            @ In, Y, a m vector of values specifying the output
                responses corresponding to the m samples specified by X
            @ In, names, an optional list of strings that specify the
                names to associate to the n input dimensions and 1
                output dimension. Default of None means input variables
                will be x0,x1...,x(n-1) and the output will be y
            @ In, graph, an optional string specifying the type of
                neighborhood graph to use. Default is 'beta skeleton,'
                but other valid types are: 'delaunay,' 'relaxed beta
                skeleton,' 'none', or 'approximate knn'
            @ In, gradient, an optional string specifying the type of
                gradient estimator to use. Currently the only available
                option is 'steepest'
            @ In, knn, an optional integer value specifying the maximum
                number of k-nearest neighbors used to begin a
                neighborhood search. In the case of graph='[relaxed]
                beta skeleton', we will begin with the specified
                approximate knn graph and prune edges that do not
                satisfy the empty region criteria.
            @ In, beta, an optional floating point value between 0 and
                2. This value is only used when graph='[relaxed] beta
                skeleton' and specifies the radius for the empty region
                graph computation (1=Gabriel graph, 2=Relative neighbor
                graph)
            @ In, normalization, an optional string specifying whether
                the inputs/output should be scaled before computing.
                Currently, two modes are supported 'zscore' and
                'feature'. 'zscore' will ensure the data has a mean of
                zero and a standard deviation of 1 by subtracting the
                mean and dividing by the variance. 'feature' scales the
                data into the unit hypercube.
            @ In, edges, an optional list of custom edges to use as a
                starting point for pruning, or in place of a computed
                graph.
            @ In, debug, an optional boolean flag for whether debugging
                output should be enabled.
        """
        super(MergeTree, self).__init__()

        self.X = X
        self.Y = Y

        self.names = names
        self.normalization = normalization

        if self.X is None or self.Y is None:
            raise ValueError('There is no data to process.')

        if self.names is None:
            self.names = []
            for d in range(self.GetDimensionality()):
                self.names.append('x%d' % d)
            self.names.append('y')

        if normalization == 'feature':
            # This doesn't work with one-dimensional arrays on older
            # versions of sklearn
            min_max_scaler = sklearn.preprocessing.MinMaxScaler()
            self.Xnorm = min_max_scaler.fit_transform(np.atleast_2d(self.X))
            self.Ynorm = min_max_scaler.fit_transform(np.atleast_2d(self.Y))
        elif normalization == 'zscore':
            self.Xnorm = sklearn.preprocessing.scale(self.X, axis=0,
                                                     with_mean=True,
                                                     with_std=True, copy=True)
            self.Ynorm = sklearn.preprocessing.scale(self.Y, axis=0,
                                                     with_mean=True,
                                                     with_std=True, copy=True)
        else:
            self.Xnorm = np.array(self.X)
            self.Ynorm = np.array(self.Y)

        if knn <= 0:
            knn = len(self.Xnorm)-1

        if debug:
            sys.stderr.write('Graph Preparation: ')
            start = time.clock()

        if edges is None:
            knnAlgorithm = sklearn.neighbors.NearestNeighbors(n_neighbors=knn,
                                                              algorithm='kd_tree')
            knnAlgorithm.fit(self.Xnorm)
            edges = knnAlgorithm.kneighbors(self.Xnorm, return_distance=False)
            if debug:
                end = time.clock()
                sys.stderr.write('%f s\n' % (end-start))

            # prevent duplicates with this guy
            pairs = []
            for e1 in range(0, edges.shape[0]):
                for col in range(0, edges.shape[1]):
                    e2 = edges.item(e1, col)
                    if e1 != e2:
                        pairs.append((e1, e2))
        else:
            pairs = edges

        # As seen here:
        #    http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
        seen = set()
        pairs = [x for x in pairs if not (x in seen or x[::-1] in seen or
                                          seen.add(x))]
        edgesToPrune = []
        for edge in pairs:
            edgesToPrune.append(edge[0])
            edgesToPrune.append(edge[1])

        if debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))
            sys.stderr.write('Decomposition: ')
            start = time.clock()

        self.__tree = MergeTreeFloat(vectorFloat(self.Xnorm.flatten()),
                                     vectorFloat(self.Y),
                                     vectorString(self.names),
                                     str(graph), str(gradient), int(knn),
                                     float(beta),
                                     vectorInt(edgesToPrune))

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

        if debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))

    def GetSampleSize(self):
        """ Returns the number of samples in the input data
            @ Out, an integer specifying the number of samples.
        """
        return len(self.Y)

    def GetDimensionality(self):
        """ Returns the dimensionality of the input space of the input
            data
            @ Out, an integer specifying the dimensionality of the input
                samples.
        """
        return self.X.shape[1]
