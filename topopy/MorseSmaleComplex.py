##############################################################################
 # Software License Agreement (BSD License)                                   #
 #                                                                            #
 # Copyright 2014 University of Utah                                          #
 # Scientific Computing and Imaging Institute                                 #
 # 72 S Central Campus Drive, Room 3750                                       #
 # Salt Lake City, UT 84112                                                   #
 #                                                                            #
 # THE BSD LICENSE                                                            #
 #                                                                            #
 # Redistribution and use in source and binary forms, with or without         #
 # modification, are permitted provided that the following conditions         #
 # are met:                                                                   #
 #                                                                            #
 # 1. Redistributions of source code must retain the above copyright          #
 #    notice, this list of conditions and the following disclaimer.           #
 # 2. Redistributions in binary form must reproduce the above copyright       #
 #    notice, this list of conditions and the following disclaimer in the     #
 #    documentation and/or other materials provided with the distribution.    #
 # 3. Neither the name of the copyright holder nor the names of its           #
 #    contributors may be used to endorse or promote products derived         #
 #    from this software without specific prior written permission.           #
 #                                                                            #
 # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       #
 # IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  #
 # OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    #
 # IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           #
 # INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   #
 # NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  #
 # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      #
 # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        #
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   #
 # THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          #
 ##############################################################################

import sys
import re
import time
import os
import itertools
import collections

import numpy as np
import sklearn.neighbors
import sklearn.linear_model
import sklearn.preprocessing

import scipy.optimize
import scipy.stats
import scipy

from .topology import AMSCFloat, vectorFloat, vectorString, vectorInt

def WeightedLinearModel(X,y,w):
  """ A wrapper for playing with the linear regression used per segment. The
      benefit of having this out here is that we do not have to adjust it in
      several places in the MorseSmaleComplex class, since it can build linear
      models for an arbitrary subset of dimensions, as well.
      @ In, X, a matrix of input samples
      @ In, y, a vector of output responses corresponding to the input samples
      @ In, w, a vector of weights corresponding to the input samples
      @ Out, a tuple consisting of the fits y-intercept and the the list of
        linear coefficients.
  """
  ## Using scipy directly to do weighted linear regression on non-centered data
  Xw = np.ones((X.shape[0],X.shape[1]+1))
  Xw[:,1:] = X
  Xw = Xw * np.sqrt(w)[:, None]
  yw = y * np.sqrt(w)
  results = scipy.linalg.lstsq(Xw, yw)[0]
  yIntercept = results[0]
  betaHat = results[1:]

  return (yIntercept,betaHat)

class MorseSmaleComplex(object):
    """ A wrapper class for the C++ approximate Morse-Smale complex Object that
        also communicates with the UI via Qt's signal interface
    """
    def __init__(self, X, Y, w=None, names=None, graph='beta skeleton',
                 gradient='steepest', knn=-1, beta=1.0, normalization=None,
                 persistence='difference', edges=None, debug=False):
        """ Initialization method that takes at minimum a set of input points and
            corresponding output responses.
            @ In, X, an m-by-n array of values specifying m n-dimensional samples
            @ In, Y, a m vector of values specifying the output responses
            corresponding to the m samples specified by X
            @ In, w, an optional m vector of values specifying the weights
            associated to each of the m samples used. Default of None means all
            points will be equally weighted
            @ In, names, an optional list of strings that specify the names to
            associate to the n input dimensions and 1 output dimension. Default of
            None means input variables will be x0,x1...,x(n-1) and the output will
            be y
            @ In, graph, an optional string specifying the type of neighborhood
            graph to use. Default is 'beta skeleton,' but other valid types are:
            'delaunay,' 'relaxed beta skeleton,' 'none', or 'approximate knn'
            @ In, gradient, an optional string specifying the type of gradient
            estimator
            to use. Currently the only available option is 'steepest'
            @ In, knn, an optional integer value specifying the maximum number of
            k-nearest neighbors used to begin a neighborhood search. In the case
            of graph='[relaxed] beta skeleton', we will begin with the specified
            approximate knn graph and prune edges that do not satisfy the empty
            region criteria.
            @ In, beta, an optional floating point value between 0 and 2. This
            value is only used when graph='[relaxed] beta skeleton' and specifies
            the radius for the empty region graph computation (1=Gabriel graph,
            2=Relative neighbor graph)
            @ In, normalization, an optional string specifying whether the
            inputs/output should be scaled before computing. Currently, two modes
            are supported 'zscore' and 'feature'. 'zscore' will ensure the data
            has a mean of zero and a standard deviation of 1 by subtracting the
            mean and dividing by the variance. 'feature' scales the data into the
            unit hypercube.
            @ In, persistence, an optional string specifying how we will compute
            the persistence hierarchy. Currently, three modes are supported
            'difference', 'probability' and 'count'. 'difference' will take the
            function value difference of the extrema and its closest function
            valued neighboring saddle, 'probability' will augment this value by
            multiplying the probability of the extremum and its saddle, and count
            will make the larger point counts more persistent.
            @ In, edges, an optional list of custom edges to use as a starting point
            for pruning, or in place of a computed graph.
            @ In, debug, an optional boolean flag for whether debugging output
            should be enabled.
        """
        super(MorseSmaleComplex,self).__init__()
        if X is not None and len(X) > 1:
            self.Build(X, Y, w, names, graph, gradient, knn, beta, normalization,
                        persistence, edges, debug)
        else:
            # Set some reasonable defaults
            self.SetEmptySettings()

    def loadData(self, filename):
        """ Opens a file and reads the data into an array, sets the data as
            an nparray and list of dimnames
            @ In, filename, string representing the data file
        """
        # Open file and read the data into an array
        # save data as nparray and list of dimnames
        fin = open(myFile)
        firstLine = fin.readline().strip()
        names = re.split(',|;| |\t', firstLine)
        data = []
        numCols = len(names)
        lineNumber = 1
        for line in fin:
            line = line.strip()
            lineNumber += 1
            if len(line) == 0:
                continue
            tokens = re.split(',|;| |\t', line)
            if numCols != len(tokens):
                errorMessage = 'Data size is inconsistent.\n\n' \
                                + 'Header line: ' + str(numCols) + ' Columns vs. ' \
                                + 'Line ' + str(lineNumber) + ': ' \
                                + str(len(tokens)) + ' Columns\n\nBad line:\n' + line
                print(errorMessage)
                throw ValueError
            dataRow = []
            for token in tokens:
                value = float(token)
                dataRow.append(value)
            if len(dataRow) == numCols:
                data.append(dataRow)
            else:
                errorMessage = 'Bad data encountered at line ' + str(lineNumber) \
                                + ':\n\nBad line:\n' + line
                print(errorMessage)
                throw ValueError
        fin.close()
        self.data = np.array(data)
        self.names = names
        ##return self.PrepareSegmentation(names, np.array(data))

    def SetEmptySettings(self):
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

        self.normalization = None

        self.names = []
        self.Xnorm = []
        self.Ynorm = []

        self.__amsc = None
        self.hierarchy = None

    def Build(self, X, Y, w=None, names=None, graph='beta skeleton', gradient='steepest', knn=-1, beta=1.0, normalization=None, persistence='difference', edges=None, debug=False):
        """ Allows the caller to basically start over with a new dataset.
            @ In, X, an m-by-n array of values specifying m n-dimensional samples
            @ In, Y, a m vector of values specifying the output responses
            corresponding to the m samples specified by X
            @ In, w, an optional m vector of values specifying the weights
            associated to each of the m samples used. Default of None means all
            points will be equally weighted
            @ In, names, an optional list of strings that specify the names to
            associate to the n input dimensions and 1 output dimension. Default of
            None means input variables will be x0,x1...,x(n-1) and the output will
            be y
            @ In, graph, an optional string specifying the type of neighborhood
            graph to use. Default is 'beta skeleton,' but other valid types are:
            'delaunay,' 'relaxed beta skeleton,' or 'approximate knn'
            @ In, gradient, an optional string specifying the type of gradient
            estimator
            to use. Currently the only available option is 'steepest'
            @ In, knn, an optional integer value specifying the maximum number of
            k-nearest neighbors used to begin a neighborhood search. In the case
            of graph='[relaxed] beta skeleton', we will begin with the specified
            approximate knn graph and prune edges that do not satisfy the empty
            region criteria.
            @ In, beta, an optional floating point value between 0 and 2. This
            value is only used when graph='[relaxed] beta skeleton' and specifies
            the radius for the empty region graph computation (1=Gabriel graph,
            2=Relative neighbor graph)
            @ In, normalization, an optional string specifying whether the
            inputs/output should be scaled before computing. Currently, two modes
            are supported 'zscore' and 'feature'. 'zscore' will ensure the data
            has a mean of zero and a standard deviation of 1 by subtracting the
            mean and dividing by the variance. 'feature' scales the data into the
            unit hypercube.
            @ In, persistence, an optional string specifying how we will compute
            the persistence hierarchy. Currently, three modes are supported
            'difference', 'probability' and 'count'. 'difference' will take the
            function value difference of the extrema and its closest function
            valued neighboring saddle, 'probability' will augment this value by
            multiplying the probability of the extremum and its saddle, and count
            will make the larger point counts more persistent.
        """
        self.SetEmptySettings()

        self.X = X
        self.Y = Y
        if w is not None:
            self.w = np.array(w)
        else:
            self.w = np.ones(len(Y))*1.0/float(len(Y))

        self.names = names
        self.normalization = normalization
        self.gradient = gradient

        if self.X is None or self.Y is None:
            # print('There is no data to process, what would the Maker have me do?')
            self.SetEmptySettings()
            return

        if self.names is None:
            self.names = []
            for d in range(self.GetDimensionality()):
                self.names.append('x%d' % d)
            self.names.append('y')

        if normalization == 'feature':
            # This doesn't work with one-dimensional arrays on older versions of
            #  sklearn
            min_max_scaler = sklearn.preprocessing.MinMaxScaler()
            self.Xnorm = min_max_scaler.fit_transform(np.atleast_2d(self.X))
            self.Ynorm = min_max_scaler.fit_transform(np.atleast_2d(self.Y))
        elif normalization == 'zscore':
            self.Xnorm = sklearn.preprocessing.scale(self.X, axis=0, with_mean=True,
                                                    with_std=True, copy=True)
            self.Ynorm = sklearn.preprocessing.scale(self.Y, axis=0, with_mean=True,
                                                    with_std=True, copy=True)
        else:
            self.Xnorm = np.array(self.X)
            self.Ynorm = np.array(self.Y)

        if knn <= 0:
            knn = len(self.Xnorm)-1

        if debug:
            sys.stderr.write('Graph Preparation: ')
            start = time.clock()

        if knn <= 0:
            knn = len(self.Y)-1

        if edges is None:
            knnAlgorithm = sklearn.neighbors.NearestNeighbors(n_neighbors=knn,
                                                                algorithm='kd_tree')
            knnAlgorithm.fit(self.Xnorm)
            edges = knnAlgorithm.kneighbors(self.Xnorm, return_distance=False)
            if debug:
                end = time.clock()
                sys.stderr.write('%f s\n' % (end-start))

            pairs = []                              # prevent duplicates with this guy
            for e1 in range(0,edges.shape[0]):
                    for col in range(0,edges.shape[1]):
                        e2 = edges.item(e1,col)
                        if e1 != e2:
                            pairs.append((e1,e2))
        else:
            pairs = edges

        # As seen here:
        #  http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
        seen = set()
        pairs = [ x for x in pairs if not (x in seen or x[::-1] in seen
                                        or seen.add(x))]
        edgeList = []
        for edge in pairs:
            edgeList.append(edge[0])
            edgeList.append(edge[1])

        if debug:
            end = time.clock()
            sys.stderr.write('%f s\n' % (end-start))
            sys.stderr.write('Decomposition: ')
            start = time.clock()

        self.__amsc = AMSCFloat(vectorFloat(self.Xnorm.flatten()),
                                vectorFloat(self.Y),
                                vectorString(self.names),
                                str(self.gradient),
                                str(persistence),
                                vectorFloat(self.w),
                                vectorInt(edgeList),
                                debug)

        if debug:
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

                self.mergeSequence[dyingIndex] = (parentIndex,p)
                self.persistences.append(p)

        self.persistences = sorted(list(set(self.persistences)))
        
        ########################################################################
        # P Save starts here.
        p_interest = 0
        # p_interest = input('Input Persistence of Interest: ')

        idx_interest = -1
        for idx in range(0,len(self.persistences)):
            if float(p_interest)<= self.persistences[idx]:
                idx_interest = idx
                break

        partitions = self.Partitions(self.persistences[idx_interest])
        cellIdxs = np.array(list(partitions.keys()))
        self.minIdxs = np.unique(cellIdxs[:,0])
        self.maxIdxs = np.unique(cellIdxs[:,1])
        for min_max_pair in list(partitions.keys()):

            if type(min_max_pair)==tuple:
                partitions[str(min_max_pair[0])+', '+str(min_max_pair[1])] = partitions[min_max_pair]
                del partitions[min_max_pair]

        self.base_partitions = partitions
        ########################################################################

    def SetWeights(self, w=None):
        """ Sets the weights associated to the m input samples
            @ In, w, optional m vector specifying the new weights to use for the
            data points. Default is None and resets the weights to be uniform.
        """
        if w is not None:
            self.w = np.array(w)
        elif len(self.Y) > 0:
            self.w = np.ones(len(self.Y))*1.0/float(len(self.Y))

    def GetMergeSequence(self):
        """ Returns a data structure holding the ordered merge sequence of extrema
            simplification
            @ Out, a dictionary of tuples where the key is the dying extrema and the
            tuple is the parent index and the persistence associated to the dying
            index, in that order.
        """
        return self.mergeSequence

    def Partitions(self, persistence=None):
        """ Returns the partitioned data based on a specified persistence level.
            @ In, persistence, a floating point value specifying the size of the
            smallest feature we want to track. Default = None means consider all
            features.
            @ Out, a dictionary lists where each key is a min-max tuple specifying
            the index of the minimum and maximum, respectively. Each entry will
            hold a list of indices specifying points that are associated to this
            min-max pair.
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
                minMax = tuple(map(int,strMinMax.split(',')))
                tupleKeyedPartitions[minMax] = indices
            self.partitions[persistence] = tupleKeyedPartitions
        return self.partitions[persistence]

    def StableManifolds(self,persistence=None):
        """ Returns the partitioned data based on a specified persistence level.
            @ In, persistence, a floating point value specifying the size of the
            smallest feature we want to track. Default = None means consider all
            features.
            @ Out, a dictionary lists where each key is a integer specifying
            the index of the maximum. Each entry will hold a list of indices
            specifying points that are associated to this maximum.
        """
        if persistence is None:
            persistence = self.persistence
        return self.__amsc.GetStableManifolds(persistence)

    def UnstableManifolds(self,persistence=None):
        """ Returns the partitioned data based on a specified persistence level.
            @ In, persistence, a floating point value specifying the size of the
            smallest feature we want to track. Default = None means consider all
            features.
            @ Out, a dictionary lists where each key is a integer specifying
            the index of the minimum. Each entry will hold a list of indices
            specifying points that are associated to this minimum.
        """
        if persistence is None:
            persistence = self.persistence
        return self.__amsc.GetUnstableManifolds(persistence)

    def Persistence(self, p=None):
        """ Sets or returns the persistence simplfication level to be used for
            representing this Morse-Smale complex
            @ In, p, a floating point value that will set the persistence value,
            if this value is set to None, then this function will return the
            current persistence leve.
            @ Out, if no p value is supplied then this function will return the
            current persistence setting. If a p value is supplied, it will be
            returned as it will be the new persistence setting of this object.
        """
        if p is None:
            return self.persistence
        self.persistence = p
        return self.persistence

    def GetNames(self):
        """ Returns the names of the input and output dimensions in the order they
            appear in the input data.
            @ Out, a list of strings specifying the input + output variable names.
        """
        return self.names

    def GetNormedX(self,rows=None,cols=None,applyFilters=False):
        """ Returns the normalized input data requested by the user
            @ In, rows, a list of non-negative integers specifying the row indices
            to return
            @ In, cols, a list of non-negative integers specifying the column
            indices to return
            @ In, applyFilters, a boolean specifying whether data filters should be
            used to prune the results
            @ Out, a matrix of floating point values specifying the normalized data
            values used in internal computations filtered by the three input
            parameters.
        """
        if rows is None:
            rows = list(range(0,self.GetSampleSize()))
        if cols is None:
            cols = list(range(0,self.GetDimensionality()))

        if applyFilters:
            rows = self.GetMask(rows)
        retValue = self.Xnorm[rows,:]
        return retValue[:,cols]

    def GetX(self, rows=None, cols=None):
        """ Returns the input data requested by the user
            @ In, rows, a list of non-negative integers specifying the row indices
            to return
            @ In, cols, a list of non-negative integers specifying the column
            indices to return
            @ Out, a matrix of floating point values specifying the input data
            values filtered by the two input parameters.
        """
        if rows is None:
            rows = list(range(0,self.GetSampleSize()))
        if cols is None:
            cols = list(range(0,self.GetDimensionality()))

        rows = sorted(list(set(rows)))

        retValue = self.X[rows,:]
        if len(rows) == 0:
            return []
        return retValue[:,cols]

    def GetY(self, indices=None):
        """ Returns the output data requested by the user
            @ In, indices, a list of non-negative integers specifying the
            row indices to return
            @ Out, a list of floating point values specifying the output data
            values filtered by the indices input parameter.
        """
        if indices is None:
            indices = list(range(0,self.GetSampleSize()))
        else:
            indices = sorted(list(set(indices)))

        if len(indices) == 0:
            return []
        return self.Y[indices]

    def GetLabel(self, indices=None):
        """ Returns the label pair indices requested by the user
            @ In, indices, a list of non-negative integers specifying the
            row indices to return
            @ Out, a list of integer 2-tuples specifying the minimum and maximum
            index of the specified rows.
        """
        if indices is None:
            indices = list(range(0,self.GetSampleSize()))
        elif isinstance(indices,collections.Iterable):
            indices = sorted(list(set(indices)))
        else:
            indices = [indices]

        if len(indices) == 0:
            return []
        partitions = self.__amsc.GetPartitions(self.persistence)
        labels = self.X.shape[0]*[None]
        for strMinMax in partitions.keys():
            partIndices = partitions[strMinMax]
            label = tuple(map(int,strMinMax.split(',')))
            for idx in np.intersect1d(partIndices,indices):
                    labels[idx] = label

        labels = np.array(labels)
        if len(indices) == 1:
            return labels[indices][0]
        return labels[indices]

    def GetWeights(self, indices=None):
        """ Returns the weights requested by the user
            @ In, indices, a list of non-negative integers specifying the
            row indices to return
            @ Out, a list of floating point values specifying the weights associated
            to the input data rows filtered by the indices input parameter.
        """
        if indices is None:
            indices = list(range(0,self.GetSampleSize()))
        else:
            indices = sorted(list(set(indices)))

        if len(indices) == 0:
            return []
        return self.w[indices]

    def GetCurrentLabels(self):
        """ Returns a list of tuples that specifies the min-max index labels
            associated to each input sample
            @ Out, a list of tuples that are each a pair of non-negative integers
            specifying the min-flow and max-flow indices associated to each input
            sample at the current level of persistence
        """
        partitions = self.Partitions(self.persistence)
        return partitions.keys()

    def GetSampleSize(self,key = None):
        """ Returns the number of samples in the input data
            @ In, key, an optional 2-tuple specifying a min-max id pair used for
            determining which partition size should be returned. If not specified
            then the size of the entire data set will be returned.
            @ Out, an integer specifying the number of samples.
        """
        if key is None:
            return len(self.Y)
        else:
            return len(self.partitions[self.persistence][key])

    def GetDimensionality(self):
        """ Returns the dimensionality of the input space of the input data
            @ Out, an integer specifying the dimensionality of the input samples.
        """
        return self.X.shape[1]

    def GetClassification(self,idx):
        """ Given an index, this function will report whether that sample is a local
            minimum, a local maximum, or a regular point.
            @ In, idx, a non-negative integer less than the sample size of the input
            data.
            @ Out, a string specifying the classification type of the input sample:
            will be 'maximum,' 'minimum,' or 'regular.'
        """
        if idx in self.minIdxs:
            return 'minimum'
        elif idx in self.maxIdxs:
            return 'maximum'
        return 'regular'

    def PrintHierarchy(self):
        """ Writes the complete Morse-Smale merge hierarchy to a string object.
            @ Out, a string object storing the entire merge hierarchy of all minima
            and maxima.
        """
        return self.__amsc.PrintHierarchy()

    def GetNeighbors(self,idx):
        """ Returns a list of neighbors for the specified index
            @ In, an integer specifying the query point
            @ Out, a integer list of neighbors indices
        """
        return self.__amsc.Neighbors(int(idx))
