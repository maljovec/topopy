##############################################################################
 # Software License Agreement (BSD License)                                   #
 #                                                                            #
 # Copyright 2018 University of Utah                                          #
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

class AnnotatedMorseSmaleComplex(MorseSmaleComplex):
    """ A wrapper class for the C++ approximate Morse-Smale complex Object that
        also communicates with the UI via Qt's signal interface
    """
    def SetEmptySettings(self):
        """
            Empties all internal storage containers
        """
        super(AnnotatedMorseSmaleComplex, self).SetEmptySettings()

        self.segmentFits = {}
        self.extremumFits = {}

        self.segmentFitnesses = {}
        self.extremumFitnesses = {}

        self.selectedExtrema = []
        self.selectedSegments = []

        self.filters = {}

    def SetWeights(self, w=None):
        """ Sets the weights associated to the m input samples
            @ In, w, optional m vector specifying the new weights to use for the
            data points. Default is None and resets the weights to be uniform.
        """
        super(AnnotatedMorseSmaleComplex, self).SetWeights(w)
        if self.FitsSynced():
            self.BuildModels()

    def SegmentFitCoefficients(self):
        """ Returns a dictionary keyed off the min-max index pairs defining
            Morse-Smale segments where the values are the linear coefficients of
            the input dimensions sorted in the same order as the input data.
            @ Out, a dictionary with tuples as keys specifying a pair of integers
            denoting minimum and maximum indices. The values associated to the
            dictionary keys are the linear coefficients fit for each min-max pair.
        """
        if self.segmentFits is None or len(self.segmentFits) == 0:
            self.BuildModels(self.persistence)
        coefficients = {}
        for key,fit in self.segmentFits.items():
            coefficients[key] = fit[1:]
            # coefficients[key] = fit[:]
        return coefficients

    def SegmentFitnesses(self):
        """ Returns a dictionary keyed off the min-max index pairs defining
            Morse-Smale segments where the values are the R^2 metrics of the linear
            fits for each Morse-Smale segment.
            @ Out, a dictionary with tuples as keys specifying a pair of integers
            denoting minimum and maximum indices. The values associated to the
            dictionary keys are the R^2 values for each linear fit of the
            Morse-Smale segments defined by the min-max pair of integers.
        """
        if self.segmentFits is None or len(self.segmentFits) == 0:
            self.BuildModels(self.persistence)
        rSquared = {}
        for key,fitness in self.segmentFitnesses.items():
            rSquared[key] = fitness
        return rSquared

    def SegmentPearsonCoefficients(self):
        """ Returns a dictionary keyed off the min-max index pairs defining
            Morse-Smale segments where the values are the Pearson correlation
            coefficients of the input dimensions sorted in the same order as the
            input data.
            @ Out, a dictionary with tuples as keys specifying a pair of integers
            denoting minimum and maximum indices. The values associated to the
            dictionary keys are the Pearson correlation coefficients associated
            to each subset of the data.
        """
        if self.segmentFits is None or len(self.segmentFits) == 0:
            self.BuildModels(self.persistence)
        pearson = {}
        for key,fit in self.pearson.items():
            pearson[key] = fit[:]
        return pearson

    def SegmentSpearmanCoefficients(self):
        """ Returns a dictionary keyed off the min-max index pairs defining
            Morse-Smale segments where the values are the Spearman rank correlation
            coefficients of the input dimensions sorted in the same order as the
            input data.
            @ Out, a dictionary with tuples as keys specifying a pair of integers
            denoting minimum and maximum indices. The values associated to the
            dictionary keys are the Spearman rank correlation coefficients
            associated to each subset of the data.
        """
        if self.segmentFits is None or len(self.segmentFits) == 0:
            self.BuildModels(self.persistence)
        spearman = {}
        for key,fit in self.spearman.items():
            spearman[key] = fit[:]
        return spearman

    def GetMask(self,indices=None):
        """ Applies all data filters to the input data and returns a list of
            filtered indices that specifies the rows of data that satisfy all
            conditions.
            @ In, indices, an optional integer list of indices to start from, if not
            supplied, then the mask will be applied to all indices of the data.
            @ Out, a 1-dimensional array of integer indices that is a subset of
            the input data row indices specifying rows that satisfy every set
            filter criterion.
        """
        if indices is None:
            indices = list(range(0,self.GetSampleSize()))
        else:
            indices = sorted(list(set(indices)))

        mask = np.ones(len(indices), dtype=bool)
        for header,bounds in self.filters.items():
            if header in self.names:
                    idx = self.names.index(header)
                    if idx >= 0 and idx < len(self.names)-1:
                        vals = self.X[indices,idx]
                    elif idx == len(self.names)-1:
                        vals = self.Y[indices]
            elif header == 'Predicted from Linear Fit':
                    vals = self.PredictY(indices, fit='linear', applyFilters=False)
            elif header == 'Predicted from Maximum Fit':
                    vals = self.PredictY(indices, fit='maximum', applyFilters=False)
            elif header == 'Predicted from Minimum Fit':
                    vals = self.PredictY(indices, fit='minimum', applyFilters=False)
            elif header == 'Residual from Linear Fit':
                    vals = self.Residuals(indices, fit='linear', applyFilters=False)
            elif header == 'Residual from Maximum Fit':
                    vals = self.Residuals(indices, fit='maximum', applyFilters=False)
            elif header == 'Residual from Minimum Fit':
                    vals = self.Residuals(indices, fit='minimum', applyFilters=False)
            elif header == 'Probability':
                    vals = self.w[indices]
            mask = np.logical_and(mask, bounds[0] <= vals)
            mask = np.logical_and(mask, vals < bounds[1])

        indices = np.array(indices)[mask]
        indices = np.array(sorted(list(set(indices))))
        return indices

    def ComputePerDimensionFitErrors(self,key):
        """ Heuristically builds lower-dimensional linear patches for a Morse-Smale
            segment specified by a tuple of integers, key. The heuristic is to sort
            the set of linear coefficients by magnitude and progressively refit the
            data using more and more dimensions and computing R^2 values for each
            lower dimensional fit until we arrive at the full dimensional linear fit
            @ In, key, a tuple of two integers specifying the minimum and maximum
            indices used to key the partition upon which we are retrieving info.
            @ Out, a tuple of three equal sized lists that specify the index order
            of the dimensions added where the indices match the input data's
            order, the R^2 values for each progressively finer fit, and the
            F-statistic for each progressively finer fit. Thus, an index order of
            [2,3,1,0] would imply the first fit uses only dimension 2, and
            the next fit uses dimension 2 and 3, and the next fit uses 2, 3, and
            1, and the final fit uses dimensions 2, 1, 3, and 0.
        """
        partitions = self.Partitions(self.persistence)
        if key not in self.segmentFits or key not in partitions:
            return None

        beta_hat = self.segmentFits[key][1:]
        yIntercept = self.segmentFits[key][0]
        # beta_hat = self.segmentFits[key][:]
        # yIntercept = 0
        items = partitions[key]

        X = self.Xnorm[np.array(items),:]
        y = self.Y[np.array(items)]
        w = self.w[np.array(items)]

        yHat = X.dot(beta_hat) + yIntercept
        RSS2 = np.sum(w*(y-yHat)**2)/np.sum(w)

        RSS1 = 0

        rSquared = []
        ## From here: http://en.wikipedia.org/wiki/F-test
        fStatistic = [] ## the computed F statistic
        indexOrder = list(reversed(np.argsort(np.absolute(beta_hat))))
        for i,nextDim in enumerate(indexOrder):
            B = np.zeros(self.GetDimensionality())
            for activeDim in indexOrder[0:(i+1)]:
                B[activeDim] = beta_hat[activeDim]

            X = self.X[np.array(items),:]
            X = X[:,indexOrder[0:(i+1)]]
            ## In the first case, X will be one-dimensional, so we have to enforce a
            ## reshape in order to get it to play nice.
            X = np.reshape(X,(len(items),i+1))
            y = self.Y[np.array(items)]
            w = self.w[np.array(items)]

            (temp_yIntercept,temp_beta_hat) = WeightedLinearModel(X,y,w)

            yHat = X.dot(temp_beta_hat) + temp_yIntercept

            # Get a weighted mean
            yMean = np.average(y,weights=w)

            RSS2 = np.sum(w*(y-yHat)**2)/np.sum(w)
            if RSS1 == 0:
                    fStatistic.append(0)
            else:
                    fStatistic.append(  (RSS1-RSS2)/(len(indexOrder)-i) \
                                    / (RSS2/(len(y)-len(indexOrder)))  )

            SStot = np.sum(w*(y-yMean)**2)/np.sum(w)
            rSquared.append(1-(RSS2/SStot))
            RSS1 = RSS2

        return (indexOrder,rSquared,fStatistic)

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
        self.segmentFits = {}
        self.extremumFits = {}
        self.segmentFitnesses = {}
        self.extremumFitnesses = {}
        return self.persistence

    def BuildModels(self,persistence=None):
        """ Forces the construction of linear fits per Morse-Smale segment and
            Gaussian fits per stable/unstable manifold for the user-specified
            persistence level.
            @ In, persistence, a floating point value specifying the simplification
            level to use, if this value is None, then we will build models based
            on the internally set persistence level for this Morse-Smale object.
        """
        self.segmentFits = {}
        self.extremumFits = {}
        self.segmentFitnesses = {}
        self.extremumFitnesses = {}
        self.BuildLinearModels(persistence)
        self.ComputeStatisticalSensitivity()

    def BuildLinearModels(self, persistence=None):
        """ Forces the construction of linear fits per Morse-Smale segment.
            @ In, persistence, a floating point value specifying the simplification
            level to use, if this value is None, then we will build models based
            on the internally set persistence level for this Morse-Smale object.
        """
        partitions = self.Partitions(persistence)

        for key,items in partitions.items():
            X = self.Xnorm[np.array(items),:]
            y = np.array(self.Y[np.array(items)])
            w = self.w[np.array(items)]

            (temp_yIntercept,temp_beta_hat) = WeightedLinearModel(X,y,w)
            self.segmentFits[key] = np.hstack((temp_yIntercept,temp_beta_hat))

            yHat = X.dot(self.segmentFits[key][1:]) + self.segmentFits[key][0]

            self.segmentFitnesses[key] = sum(np.sqrt((yHat-y)**2))

    def GetLabel(self, indices=None, applyFilters=False):
        """ Returns the label pair indices requested by the user
            @ In, indices, a list of non-negative integers specifying the
            row indices to return
            @ In, applyFilters, a boolean specifying whether data filters should be
            used to prune the results
            @ Out, a list of integer 2-tuples specifying the minimum and maximum
            index of the specified rows.
        """
        if applyFilters:
            indices = self.GetMask(indices)
        return super(AnnotatedMorseSmaleComplex, self).GetLabel(indices)

    def GetNormedX(self, rows=None, cols=None, applyFilters=False):
        """ Returns the normalized input data requested by the user
            @ In, rows, a list of non-negative integers specifying the row
            indices to return
            @ In, cols, a list of non-negative integers specifying the column
            indices to return
            @ In, applyFilters, a boolean specifying whether data filters should be
            used to prune the results
            @ Out, a matrix of floating point values specifying the normalized data
            values used in internal computations filtered by the three input
            parameters.
        """
        if applyFilters:
            rows = self.GetMask(rows)

        return super(AnnotatedMorseSmaleComplex, self).GetNormedX(rows, cols)

    def GetSelectedExtrema(self):
        """ Returns the extrema highlighted as being selected in an attached UI
            @ Out, a list of non-negative integer indices specifying the extrema
            selected.
        """
        return self.selectedExtrema

    def GetSelectedSegments(self):
        """ Returns the Morse-Smale segments highlighted as being selected in an
            attached UI
            @ Out, a list of non-negative integer index pairs specifying the min-max
            pairs associated to the selected Morse-Smale segments.
        """
        return self.selectedSegments

    def GetWeights(self, indices=None, applyFilters=False):
        """ Returns the weights requested by the user
            @ In, indices, a list of non-negative integers specifying the
            row indices to return
            @ In, applyFilters, a boolean specifying whether data filters should be
            used to prune the results
            @ Out, a list of floating point values specifying the weights associated
            to the input data rows filtered by the two input parameters.
        """
        if applyFilters:
            indices = self.GetMask(indices)

        return super(AnnotatedMorseSmaleComplex, self).GetWeights(indices)

    def GetX(self, rows=None, cols=None, applyFilters=False):
        """ Returns the input data requested by the user
            @ In, rows, a list of non-negative integers specifying the row indices
            to return
            @ In, cols, a list of non-negative integers specifying the column
            indices to return
            @ In, applyFilters, a boolean specifying whether data filters should be
            used to prune the results
            @ Out, a matrix of floating point values specifying the input data
            values filtered by the three input parameters.
        """
        if applyFilters:
            rows = self.GetMask(rows)

        return super(AnnotatedMorseSmaleComplex, self).GetX(rows, cols)

    def GetY(self, indices=None, applyFilters=False):
        """ Returns the output data requested by the user
            @ In, indices, a list of non-negative integers specifying the
            row indices to return
            @ In, applyFilters, a boolean specifying whether data filters should be
            used to prune the results
            @ Out, a list of floating point values specifying the output data
            values filtered by the two input parameters.
        """
        if applyFilters:
            indices = self.GetMask(indices)

        return super(AnnotatedMorseSmaleComplex, self).GetY(indices)

    def Predict(self, x, key):
        """ Returns the predicted response of x given a model index
            @ In, x, a list of input values matching the dimensionality of the
            input space
            @ In, key, a 2-tuple specifying a min-max id pair used for determining
            which model is being used for prediction
            @ Out, a predicted response value for the given input point
        """
        partitions = self.Partitions(self.persistence)
        beta_hat = self.segmentFits[key][1:]
        y_intercept = self.segmentFits[key][0]
        if len(x.shape) == 1:
            return x.dot(beta_hat) + y_intercept
        else:
            predictions = []
            for xi in x:
                predictions.append(xi.dot(beta_hat) + y_intercept)
            return predictions

    def PredictY(self,indices=None, fit='linear',applyFilters=False):
        """ Returns the predicted output values requested by the user
            @ In, indices, a list of non-negative integers specifying the
            row indices to predict
            @ In, fit, an optional string specifying which fit should be used to
            predict each location, 'linear' = Morse-Smale segment, 'maxima' =
            descending/stable manifold, 'minima' = ascending/unstable manifold.
            Only 'linear' is available in this version.
            @ In, applyFilters, a boolean specifying whether data filters should be
            used to prune the results
            @ Out, a list of floating point values specifying the predicted output
            values filtered by the three input parameters.
        """
        partitions = self.Partitions(self.persistence)

        predictedY = np.zeros(self.GetSampleSize())
        if fit == 'linear':
            for key,items in partitions.items():
                    beta_hat = self.segmentFits[key][1:]
                    y_intercept = self.segmentFits[key][0]
                    for idx in items:
                        predictedY[idx] = self.Xnorm[idx,:].dot(beta_hat) + y_intercept
        ## Possible extension to fit data per stable or unstable manifold would
        ## go here

        if indices is None:
            indices = list(range(0,self.GetSampleSize()))
        if applyFilters:
            indices = self.GetMask(indices)
        indices = np.array(sorted(list(set(indices))))
        return predictedY[indices]

    def Residuals(self, indices=None, fit='linear', signed=False, applyFilters=False):
        """ Returns the residual between the output data and the predicted output
            values requested by the user
            @ In, indices, a list of non-negative integers specifying the
            row indices for which to compute residuals
            @ In, fit, an optional string specifying which fit should be used to
            predict each location, 'linear' = Morse-Smale segment, 'maxima' =
            descending/stable manifold, 'minima' = ascending/unstable manifold
            @ In, applyFilters, a boolean specifying whether data filters should be
            used to prune the results
            @ Out, a list of floating point values specifying the signed difference
            between the predicted output values and the original output data
            filtered by the three input parameters.
        """
        if indices is None:
            indices = list(range(0,self.GetSampleSize()))
        else:
            indices = sorted(list(set(indices)))
            
        if applyFilters:
            indices = self.GetMask(indices)

        indices = np.array(sorted(list(set(indices))))

        yRange = max(self.Y) - min(self.Y)
        actualY = self.GetY(indices)
        predictedY = self.PredictY(indices,fit)
        if signed:
            residuals = (actualY-predictedY)/yRange
        else:
            residuals = np.absolute(actualY-predictedY)/yRange

        return residuals

    def ComputeStatisticalSensitivity(self):
        """ Computes the per segment Pearson correlation coefficients and the
            Spearman rank correlation coefficients and stores them internally.
        """
        partitions = self.Partitions()

        self.pearson = {}
        self.spearman = {}
        for key,items in partitions.items():
            X = self.Xnorm[np.array(items),:]
            y = self.Y[np.array(items)]

            self.pearson[key] = []
            self.spearman[key] = []

            for col in range(0,X.shape[1]):
                    sigmaXcol = np.std(X[:,col])
                    self.pearson[key].append(scipy.stats.pearsonr(X[:,col], y)[0])
                    self.spearman[key].append(scipy.stats.spearmanr(X[:,col], y)[0])
