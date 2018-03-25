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
import collections
import json
import warnings

import numpy as np
import sklearn.preprocessing

import nglpy

from .topology import AMSCFloat, vectorFloat, vectorString


class MorseSmaleComplex(object):
    """ A wrapper class for the C++ approximate Morse-Smale complex
        Object
    """
    def __init__(self, graph='beta skeleton', gradient='steepest',
                 max_neighbors=-1, beta=1.0, normalization=None,
                 simplification='difference', connect=False, aggregator=None,
                 debug=False):
        """ Initialization method that takes at minimum a set of input
            points and corresponding output responses.
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
            @ In, simplification, an optional string specifying how we
            will compute the simplification hierarchy. Currently, three
            modes are supported 'difference', 'probability' and 'count'.
            'difference' will take the function value difference of the
            extrema and its closest function valued neighboring saddle
            (standard persistence simplification), 'probability' will
            augment this value by multiplying the probability of the
            extremum and its saddle, and count will order the
            simplification by the size (number of points) in each
            manifold such that smaller features will be absorbed into
            neighboring larger features first.
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
        super(MorseSmaleComplex, self).__init__()
        self.reset()

        self.graph = graph
        self.gradient = gradient
        self.max_neighbors = max_neighbors
        self.beta = beta
        self.simplification = simplification
        self.normalization = normalization
        self.gradient = gradient
        self.connect = connect
        self.debug = debug
        self.aggregator = aggregator

        # This feature is for controlling how many decimal places of
        # precision will be used to determine if two points should
        # be considered the same
        self.precision = 15

    def reset(self):
        """
            Empties all internal storage containers
        """
        self.persistences = []

        self.partitions = {}
        self.base_partitions = {}
        self.persistence = 0.

        self.mergeSequence = {}

        self.minIdxs = []
        self.maxIdxs = []

        self.X = []
        self.Y = []
        self.w = []

        self.names = []
        self.Xnorm = []
        self.Ynorm = []

        self.__amsc = None
        self.hierarchy = None

    def __set_data(self, X, Y, w=None, names=None):
        """ Internally assigns the input data and normalizes it
            according to the user's specifications
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
        """
        self.X = X
        self.Y = Y
        self.check_duplicates()

        if w is not None:
            self.w = np.array(w)
        else:
            self.w = np.ones(len(Y))*1.0/float(len(Y))

        self.names = names

        if self.names is None:
            self.names = []
            for d in range(self.get_dimensionality()):
                self.names.append('x%d' % d)
            self.names.append('y')

        if self.normalization == 'feature':
            # This doesn't work with one-dimensional arrays on older
            # versions of sklearn
            min_max_scaler = sklearn.preprocessing.MinMaxScaler()
            self.Xnorm = min_max_scaler.fit_transform(np.atleast_2d(self.X))
            self.Ynorm = min_max_scaler.fit_transform(np.atleast_2d(self.Y))
        elif self.normalization == 'zscore':
            self.Xnorm = sklearn.preprocessing.scale(self.X,
                                                     axis=0,
                                                     with_mean=True,
                                                     with_std=True,
                                                     copy=True)
            self.Ynorm = sklearn.preprocessing.scale(self.Y,
                                                     axis=0,
                                                     with_mean=True,
                                                     with_std=True,
                                                     copy=True)
        else:
            self.Xnorm = np.array(self.X)
            self.Ynorm = np.array(self.Y)

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
        self.reset()

        if X is None or Y is None:
            return

        self.__set_data(X, Y, w, names)

        if self.debug:
            sys.stderr.write('Graph Preparation: ')
            start = time.clock()

        graph_rep = nglpy.Graph(self.Xnorm, self.graph, self.max_neighbors,
                                self.beta, connect=self.connect)

        if self.debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))
            sys.stderr.write('Decomposition: ')
            start = time.clock()

        self.__amsc = AMSCFloat(vectorFloat(self.Xnorm.flatten()),
                                vectorFloat(self.Y),
                                vectorString(self.names),
                                str(self.gradient),
                                str(self.simplification),
                                vectorFloat(self.w),
                                graph_rep.full_graph(),
                                self.debug)

        if self.debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))

        hierarchy = self.__amsc.PrintHierarchy().strip().split(' ')
        self.hierarchy = hierarchy
        self.persistences = []
        self.mergeSequence = {}
        for line in hierarchy:
            if line.startswith('Maxima') or line.startswith('Minima'):
                tokens = line.split(',')
                p = float(tokens[1])
                dyingIndex = int(tokens[2])
                parentIndex = int(tokens[3])
                saddleIndex = int(tokens[4])

                self.mergeSequence[dyingIndex] = (p, parentIndex, saddleIndex)
                self.persistences.append(p)

        self.persistences = sorted(list(set(self.persistences)))

        ################################################################
        # P Save starts here.
        p_interest = 0
        # p_interest = input('Input Persistence of Interest: ')
        # idx_interest = -1
        # for idx in range(0,len(self.persistences)):
        #     if float(p_interest)<= self.persistences[idx]:
        #         idx_interest = idx
        #         break

        partitions = self.get_partitions(p_interest)
        cellIdxs = np.array(list(partitions.keys()))
        self.minIdxs = np.unique(cellIdxs[:, 0])
        self.maxIdxs = np.unique(cellIdxs[:, 1])

        # Does this need to be stored as a string? This seems excessive.
        for min_max_pair in list(partitions.keys()):
            new_key = str(min_max_pair[0])+', '+str(min_max_pair[1])
            partitions[new_key] = partitions[min_max_pair]
            del partitions[min_max_pair]

        self.base_partitions = partitions

        ################################################################

    def load_data_and_build(self, filename, delimiter=','):
        """ Convenience function for directly working with a data file.
            This opens a file and reads the data into an array, sets the
            data as an nparray and list of dimnames
            @ In, filename, string representing the data file
        """
        data = np.genfromtxt(filename, dtype=float, delimiter=delimiter,
                             names=True)
        names = list(data.dtype.names)
        data = data.view(np.float64).reshape(data.shape + (-1,))

        X = data[:, 0:-1]
        Y = data[:, -1]

        self.build(X=X, Y=Y, names=names)

    def save(self, hierarchyFilename=None, partitionFilename=None):
        """ Saves a constructed Morse-Smale Complex in json file
            @ In, hierarchyFilename, a filename for storing the
            hierarchical merging of features
            @ In, partitionFilename, a filename for storing the base
            level partitions in the data
        """
        if partitionFilename is None:
            partitionFilename = 'Base_Partition.json'
        with open(partitionFilename, 'w') as fp:
            json.dump(self.base_partitions, fp)
            fp.close()

        if hierarchyFilename is None:
            hierarchyFilename = 'Hierarchy.csv'
        with open(hierarchyFilename, 'w') as modified:
            for line in self.hierarchy:
                tokens = line.split(',')
                if (tokens[0] == 'Maxima'):
                    modified.write(tokens[1] + ',1,' + tokens[2] + ',' +
                                   tokens[3] + ',' + tokens[4] + '\n')
                else:
                    modified.write(tokens[1] + ',0,' + tokens[2] + ',' +
                                   tokens[3] + ',' + tokens[4] + '\n')
            modified.close()

    # Depending on the persistence simplification strategy, this could
    # alter the hierarchy, so let's remove this feature until further
    # notice, also the weighting feature is still pretty experimental:

    # def set_weights(self, w=None):
    #     """ Sets the weights associated to the m input samples
    #         @ In, w, optional m vector specifying the new weights to
    #         use for the data points. Default is None and resets the
    #         weights to be uniform.
    #     """
    #     if w is not None:
    #         self.w = np.array(w)
    #     elif len(self.Y) > 0:
    #         self.w = np.ones(len(self.Y))*1.0/float(len(self.Y))

    def get_merge_sequence(self):
        """ Returns a data structure holding the ordered merge sequence
            of extrema simplification
            @ Out, a dictionary of tuples where the key is the dying
            extrema and the tuple is the the persistence, parent index,
            and the saddle index associated to the dying index, in that
            order.
        """
        return self.mergeSequence

    def get_partitions(self, persistence=None):
        """ Returns the partitioned data based on a specified
            persistence level.
            @ In, persistence, a floating point value specifying the
            size of the smallest feature we want to track.
            Default = None means consider all features.
            @ Out, a dictionary lists where each key is a min-max tuple
            specifying the index of the minimum and maximum,
            respectively. Each entry will hold a list of indices
            specifying points that are associated to this min-max pair.
        """
        if self.__amsc is None:
            return None
        if persistence is None:
            persistence = self.persistence
        if persistence not in self.partitions:
            partitions = self.__amsc.GetPartitions(persistence)
            tupleKeyedPartitions = {}
            minMaxKeys = partitions.keys()
            for strMinMax in minMaxKeys:
                indices = partitions[strMinMax]
                minMax = tuple(map(int, strMinMax.split(',')))
                tupleKeyedPartitions[minMax] = indices
            self.partitions[persistence] = tupleKeyedPartitions
            # self.partitions[persistence] = partitions
        return self.partitions[persistence]

    def get_stable_manifolds(self, persistence=None):
        """ Returns the partitioned data based on a specified
            persistence level.
            @ In, persistence, a floating point value specifying the
            size of the smallest feature we want to track.
            Default = None means consider all features.
            @ Out, a dictionary lists where each key is a integer
            specifying the index of the maximum. Each entry will hold a
            list of indices specifying points that are associated to
            this maximum.
        """
        if persistence is None:
            persistence = self.persistence
        return self.__amsc.GetStableManifolds(persistence)

    def get_unstable_manifolds(self, persistence=None):
        """ Returns the partitioned data based on a specified
            persistence level.
            @ In, persistence, a floating point value specifying the
            size of the smallest feature we want to track.
            Default = None means consider all features.
            @ Out, a dictionary lists where each key is a integer
            specifying the index of the minimum. Each entry will hold a
            list of indices specifying points that are associated to
            this minimum.
        """
        if persistence is None:
            persistence = self.persistence
        return self.__amsc.GetUnstableManifolds(persistence)

    def get_persistence(self):
        """ Sets the persistence simplfication level to be
            used for representing this Morse-Smale complex
            @ Out, floating point value specifying the current
            persistence setting.
        """
        return self.persistence

    def set_persistence(self, p):
        """ Sets the persistence simplfication level to be
            used for representing this Morse-Smale complex
            @ In, p, a floating point value that will set the
            persistence value
        """
        self.persistence = p

    def get_names(self):
        """ Returns the names of the input and output dimensions in the
            order they appear in the input data.
            @ Out, a list of strings specifying the input + output
            variable names.
        """
        return self.names

    def get_normed_x(self, rows=None, cols=None):
        """ Returns the normalized input data requested by the user
            @ In, rows, a list of non-negative integers specifying the
            row indices to return
            @ In, cols, a list of non-negative integers specifying the
            column indices to return
            @ Out, a matrix of floating point values specifying the
            normalized data values used in internal computations
            filtered by the three input parameters.
        """
        if rows is None:
            rows = list(range(0, self.get_sample_size()))
        if cols is None:
            cols = list(range(0, self.get_dimensionality()))

        if not hasattr(rows, '__iter__'):
            rows = [rows]
        rows = sorted(list(set(rows)))

        retValue = self.Xnorm[rows, :]
        return retValue[:, cols]

    def get_x(self, rows=None, cols=None):
        """ Returns the input data requested by the user
            @ In, rows, a list of non-negative integers specifying the
            row indices to return
            @ In, cols, a list of non-negative integers specifying the
            column indices to return
            @ Out, a matrix of floating point values specifying the
            input data values filtered by the two input parameters.
        """
        if rows is None:
            rows = list(range(0, self.get_sample_size()))
        if cols is None:
            cols = list(range(0, self.get_dimensionality()))

        if not hasattr(rows, '__iter__'):
            rows = [rows]
        rows = sorted(list(set(rows)))

        retValue = self.X[rows, :]
        if len(rows) == 0:
            return []
        return retValue[:, cols]

    def get_y(self, indices=None):
        """ Returns the output data requested by the user
            @ In, indices, a list of non-negative integers specifying
            the row indices to return
            @ Out, an nparray of floating point values specifying the output
            data values filtered by the indices input parameter.
        """
        if indices is None:
            indices = list(range(0, self.get_sample_size()))
        else:
            if not hasattr(indices, '__iter__'):
                indices = [indices]
            indices = sorted(list(set(indices)))

        if len(indices) == 0:
            return []
        return self.Y[indices]

    def get_label(self, indices=None):
        """ Returns the label pair indices requested by the user
            @ In, indices, a list of non-negative integers specifying
            the row indices to return
            @ Out, a list of integer 2-tuples specifying the minimum and
            maximum index of the specified rows.
        """
        if indices is None:
            indices = list(range(0, self.get_sample_size()))
        elif isinstance(indices, collections.Iterable):
            indices = sorted(list(set(indices)))
        else:
            indices = [indices]

        if len(indices) == 0:
            return []
        partitions = self.__amsc.GetPartitions(self.persistence)
        labels = self.X.shape[0]*[None]
        for strMinMax in partitions.keys():
            partIndices = partitions[strMinMax]
            label = tuple(map(int, strMinMax.split(',')))
            for idx in np.intersect1d(partIndices, indices):
                    labels[idx] = label

        labels = np.array(labels)
        if len(indices) == 1:
            return labels[indices][0]
        return labels[indices]

    def get_weights(self, indices=None):
        """ Returns the weights requested by the user
            @ In, indices, a list of non-negative integers specifying
            the row indices to return
            @ Out, a list of floating point values specifying the
            weights associated to the input data rows filtered by the
            indices input parameter.
        """
        if indices is None:
            indices = list(range(0, self.get_sample_size()))
        else:
            indices = sorted(list(set(indices)))

        if len(indices) == 0:
            return []
        return self.w[indices]

    def get_current_labels(self):
        """ Returns a list of tuples that specifies the min-max index
            labels associated to each input sample
            @ Out, a list of tuples that are each a pair of non-negative
            integers specifying the min-flow and max-flow indices
            associated to each input sample at the current level of
            persistence
        """
        partitions = self.get_partitions(self.persistence)
        return partitions.keys()

    def get_sample_size(self, key=None):
        """ Returns the number of samples in the input data
            @ In, key, an optional 2-tuple specifying a min-max id pair
            used for determining which partition size should be
            returned. If not specified then the size of the entire data
            set will be returned.
            @ Out, an integer specifying the number of samples.
        """
        if key is None:
            return len(self.Y)
        else:
            return len(self.partitions[self.persistence][key])

    def get_dimensionality(self):
        """ Returns the dimensionality of the input space of the input
            data
            @ Out, an integer specifying the dimensionality of the input
            samples.
        """
        return self.X.shape[1]

    def get_classification(self, idx):
        """ Given an index, this function will report whether that
            sample is a local minimum, a local maximum, or a regular
            point.
            @ In, idx, a non-negative integer less than the sample size
            of the input data.
            @ Out, a string specifying the classification type of the
            input sample: will be 'maximum,' 'minimum,' or 'regular.'
        """
        if idx in self.minIdxs:
            return 'minimum'
        elif idx in self.maxIdxs:
            return 'maximum'
        return 'regular'

    def print_hierarchy(self):
        """ Writes the complete Morse-Smale merge hierarchy to a string
            object.
            @ Out, a string object storing the entire merge hierarchy of
            all minima and maxima.
        """
        return self.__amsc.PrintHierarchy()

    def get_neighbors(self, idx):
        """ Returns a list of neighbors for the specified index
            @ In, an integer specifying the query point
            @ Out, a integer list of neighbors indices
        """
        return self.__amsc.Neighbors(int(idx))

    def collapse_duplicates(self):
        if self.aggregator is None:
            return

        if 'min' in self.aggregator.lower():
            aggregator = np.min
        elif 'max' in self.aggregator.lower():
            aggregator = np.max
        elif 'median' in self.aggregator.lower():
            aggregator = np.median
        elif self.aggregator.lower() in ['average', 'mean']:
            aggregator = np.mean
        else:
            warnings.warn('Aggregator \"{}\" not understood. Skipping ' +
                          'sample aggregation.'.format(self.aggregator))

        X = self.X.round(decimals=self.precision)

        unique_xs = np.unique(X, axis=0)

        old_size = len(X)
        new_size = len(unique_xs)
        if old_size == new_size:
            return

        reduced_y = np.empty(new_size)

        warnings.warn('Domain space duplicates caused a data reduction. ' +
                      'Original size: {} vs. New size: {}'.format(old_size,
                                                                  new_size))

        for i, distinct_row in enumerate(unique_xs):
            filtered_rows = np.all(X == distinct_row, axis=1)
            reduced_y[i] = aggregator(self.Y[filtered_rows])

        self.X = unique_xs
        self.Y = reduced_y
        return unique_xs, reduced_y

    def check_duplicates(self):
        """ Function to test whether duplicates exist in the input or
            output space. First, if an aggregator function has been
            specified, the domain space duplicates will be consolidated
            using the function to generate a new range value for that
            shared point. Otherwise, it will raise a ValueError.
            The function will raise a warning if duplicates exist in the
            output space
            @Out, None
        """
        self.collapse_duplicates()
        unique_ys = len(np.unique(self.Y, axis=0))
        unique_xs, indices = np.unique(self.X.round(decimals=self.precision),
                                       axis=0, return_index=True)
        unique_xs = len(unique_xs)

        if len(self.Y) != unique_ys:
            warnings.warn('Range space has duplicates. Simulation of ' +
                          'simplicity may help, but artificial noise may ' +
                          'occur in flat regions of the domain. Sample size:' +
                          '{} vs. Unique Records: {}'.format(len(self.Y),
                                                             unique_ys))

        if len(self.X) != unique_xs:
            raise ValueError('Domain space has duplicates. Try using an ' +
                             'aggregator function to consolidate duplicates ' +
                             'into a single sample with one range value. ' +
                             'e.g., MorseSmaleComplex(aggregator=\'max\'). ' +
                             '\n\tNumber of ' +
                             'Records: {}\n\tNumber of Unique Records: {}\n'
                             .format(len(self.X), unique_xs))
