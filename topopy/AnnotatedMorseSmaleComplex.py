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

import numpy as np
from scipy import stats
from scipy import linalg
from .MorseSmaleComplex import MorseSmaleComplex


def weighted_linear_model(X, y, w):
    """ A wrapper for playing with the linear regression used per
        segment. The benefit of having this out here is that we do not
        have to adjust it in several places in the MorseSmaleComplex
        class, since it can build linear models for an arbitrary subset
        of dimensions, as well.
      @ In, X, a matrix of input samples
      @ In, y, a vector of output responses corresponding to the input
      samples
      @ In, w, a vector of weights corresponding to the input samples
      @ Out, a tuple consisting of the fits y-intercept and the the list
      of linear coefficients.
    """
    # Using scipy directly to do weighted linear regression on
    # non-centered data
    Xw = np.ones((X.shape[0], X.shape[1]+1))
    Xw[:, 1:] = X
    Xw = Xw * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    results = linalg.lstsq(Xw, yw)[0]
    yIntercept = results[0]
    betaHat = results[1:]

    return (yIntercept, betaHat)


class AnnotatedMorseSmaleComplex(MorseSmaleComplex):
    """
        A wrapper class for the C++ approximate Morse-Smale complex
        Object that also communicates with the UI via Qt's signal
        interface
    """
    def reset(self):
        """
            Empties all internal storage containers
        """
        super(AnnotatedMorseSmaleComplex, self).reset()

        self.segmentFits = {}
        self.extremumFits = {}

        self.segmentFitnesses = {}
        self.extremumFitnesses = {}

        self.selectedExtrema = []
        self.selectedSegments = []

        self.filters = {}

    def set_weights(self, w=None):
        """ Sets the weights associated to the m input samples
            @ In, w, optional m vector specifying the new weights to use
            for the data points. Default is None and resets the weights
            to be uniform.
        """
        super(AnnotatedMorseSmaleComplex, self).set_weights(w)
        if self.fits_synced():
            self.build_models()

    def segment_fit_coefficients(self):
        """ Returns a dictionary keyed off the min-max index pairs
            defining Morse-Smale segments where the values are the
            linear coefficients of the input dimensions sorted in the
            same order as the input data.
            @ Out, a dictionary with tuples as keys specifying a pair of
            integers denoting minimum and maximum indices. The values
            associated to the dictionary keys are the linear
            coefficients fit for each min-max pair.
        """
        if self.segmentFits is None or len(self.segmentFits) == 0:
            self.build_models(self.persistence)
        coefficients = {}
        for key, fit in self.segmentFits.items():
            coefficients[key] = fit[1:]
            # coefficients[key] = fit[:]
        return coefficients

    def segment_fitnesses(self):
        """ Returns a dictionary keyed off the min-max index pairs
            defining Morse-Smale segments where the values are the R^2
            metrics of the linear fits for each Morse-Smale segment.
            @ Out, a dictionary with tuples as keys specifying a pair of
            integers denoting minimum and maximum indices. The values
            associated to the dictionary keys are the R^2 values for
            each linear fit of the Morse-Smale segments defined by the
            min-max pair of integers.
        """
        if self.segmentFits is None or len(self.segmentFits) == 0:
            self.build_models(self.persistence)
        rSquared = {}
        for key, fitness in self.segmentFitnesses.items():
            rSquared[key] = fitness
        return rSquared

    def segment_pearson_coefficients(self):
        """ Returns a dictionary keyed off the min-max index pairs
            defining Morse-Smale segments where the values are the
            Pearson correlation coefficients of the input dimensions
            sorted in the same order as the input data.
            @ Out, a dictionary with tuples as keys specifying a pair of
            integers denoting minimum and maximum indices. The values
            associated to the dictionary keys are the Pearson
            correlation coefficients associated to each subset of the
            data.
        """
        if self.segmentFits is None or len(self.segmentFits) == 0:
            self.build_models(self.persistence)
        pearson = {}
        for key, fit in self.pearson.items():
            pearson[key] = fit[:]
        return pearson

    def segment_spearman_coefficients(self):
        """ Returns a dictionary keyed off the min-max index pairs
            defining Morse-Smale segments where the values are the
            Spearman rank correlation coefficients of the input
            dimensions sorted in the same order as the input data.
            @ Out, a dictionary with tuples as keys specifying a pair of
            integers denoting minimum and maximum indices. The values
            associated to the dictionary keys are the Spearman rank
            correlation coefficients associated to each subset of the
            data.
        """
        if self.segmentFits is None or len(self.segmentFits) == 0:
            self.build_models(self.persistence)
        spearman = {}
        for key, fit in self.spearman.items():
            spearman[key] = fit[:]
        return spearman

    def get_mask(self, indices=None):
        """
            Applies all data filters to the input data and returns a
            list of filtered indices that specifies the rows of data
            that satisfy all conditions.
            @ In, indices, an optional integer list of indices to start
            from, if not supplied, then the mask will be applied to all
            indices of the data.
            @ Out, a 1-dimensional array of integer indices that is a
            subset of the input data row indices specifying rows that
            satisfy every set filter criterion.
        """
        if indices is None:
            indices = list(range(0, self.get_sample_size()))
        else:
            indices = sorted(list(set(indices)))

        mask = np.ones(len(indices), dtype=bool)
        for header, bounds in self.filters.items():
            if header in self.names:
                    idx = self.names.index(header)
                    if idx >= 0 and idx < len(self.names)-1:
                        vals = self.X[indices, idx]
                    elif idx == len(self.names)-1:
                        vals = self.Y[indices]
            elif header == 'Predicted from Linear Fit':
                    vals = self.predict_y(indices, fit='linear',
                                          applyFilters=False)
            elif header == 'Predicted from Maximum Fit':
                    vals = self.predict_y(indices, fit='maximum',
                                          applyFilters=False)
            elif header == 'Predicted from Minimum Fit':
                    vals = self.predict_y(indices, fit='minimum',
                                          applyFilters=False)
            elif header == 'Residual from Linear Fit':
                    vals = self.residuals(indices, fit='linear',
                                          applyFilters=False)
            elif header == 'Residual from Maximum Fit':
                    vals = self.residuals(indices, fit='maximum',
                                          applyFilters=False)
            elif header == 'Residual from Minimum Fit':
                    vals = self.residuals(indices, fit='minimum',
                                          applyFilters=False)
            elif header == 'Probability':
                    vals = self.w[indices]
            mask = np.logical_and(mask, bounds[0] <= vals)
            mask = np.logical_and(mask, vals < bounds[1])

        indices = np.array(indices)[mask]
        indices = np.array(sorted(list(set(indices))))
        return indices

    def compute_per_dimension_fit_errors(self, key):
        """
            Heuristically builds lower-dimensional linear patches for a
            Morse-Smale segment specified by a tuple of integers, key.
            The heuristic is to sort the set of linear coefficients by
            magnitude and progressively refit the data using more and
            more dimensions and computing R^2 values for each lower
            dimensional fit until we arrive at the full dimensional
            linear fit
            @ In, key, a tuple of two integers specifying the minimum
            and maximum indices used to key the partition upon which we
            are retrieving info.
            @ Out, a tuple of three equal sized lists that specify the
            index order of the dimensions added where the indices match
            the input data's order, the R^2 values for each
            progressively finer fit, and the F-statistic for each
            progressively finer fit. Thus, an index order of [2,3,1,0]
            would imply the first fit uses only dimension 2, and the
            next fit uses dimension 2 and 3, and the next fit uses 2, 3,
            and 1, and the final fit uses dimensions 2, 1, 3, and 0.
        """
        partitions = self.get_partitions(self.persistence)
        if key not in self.segmentFits or key not in partitions:
            return None

        beta_hat = self.segmentFits[key][1:]
        yIntercept = self.segmentFits[key][0]
        # beta_hat = self.segmentFits[key][:]
        # yIntercept = 0
        items = partitions[key]

        X = self.Xnorm[np.array(items), :]
        y = self.Y[np.array(items)]
        w = self.w[np.array(items)]

        yHat = X.dot(beta_hat) + yIntercept
        RSS2 = np.sum(w*(y-yHat)**2)/np.sum(w)

        RSS1 = 0

        rSquared = []
        # the computed F statistic
        # From here: http://en.wikipedia.org/wiki/F-test
        fStatistic = []
        indexOrder = list(reversed(np.argsort(np.absolute(beta_hat))))
        for i, _ in enumerate(indexOrder):
            B = np.zeros(self.get_dimensionality())
            for activeDim in indexOrder[0:(i+1)]:
                B[activeDim] = beta_hat[activeDim]

            X = self.X[np.array(items), :]
            X = X[:, indexOrder[0:(i+1)]]
            # In the first case, X will be one-dimensional, so we have
            # to enforce a reshape in order to get it to play nice.
            X = np.reshape(X, (len(items), i+1))
            y = self.Y[np.array(items)]
            w = self.w[np.array(items)]

            (temp_yIntercept, temp_beta_hat) = weighted_linear_model(X, y, w)

            yHat = X.dot(temp_beta_hat) + temp_yIntercept

            # Get a weighted mean
            yMean = np.average(y, weights=w)

            RSS2 = np.sum(w*(y-yHat)**2)/np.sum(w)
            if RSS1 == 0:
                    fStatistic.append(0)
            else:
                    fStatistic.append((RSS1-RSS2)/(len(indexOrder)-i) /
                                      (RSS2/(len(y)-len(indexOrder))))

            SStot = np.sum(w*(y-yMean)**2)/np.sum(w)
            rSquared.append(1-(RSS2/SStot))
            RSS1 = RSS2

        return (indexOrder, rSquared, fStatistic)

    def get_persistence(self):
        """ Returns the persistence simplfication level to be used for
            representing this Morse-Smale complex
            @ Out, floating point value representing the current
            persistence setting.
        """
        return self.persistence

    def set_persistence(self, p):
        """ Sets the persistence simplfication level to be used for
            representing this Morse-Smale complex
            @ In, p, a floating point value that will set the
            persistence level
        """
        self.persistence = p
        self.segmentFits = {}
        self.extremumFits = {}
        self.segmentFitnesses = {}
        self.extremumFitnesses = {}

    def build_models(self, persistence=None):
        """ Forces the construction of linear fits per Morse-Smale
            segment and Gaussian fits per stable/unstable manifold for
            the user-specified persistence level.
            @ In, persistence, a floating point value specifying the
            simplification level to use, if this value is None, then we
            will build models based on the internally set persistence
            level for this Morse-Smale object.
        """
        self.segmentFits = {}
        self.extremumFits = {}
        self.segmentFitnesses = {}
        self.extremumFitnesses = {}
        self.build_linear_models(persistence)
        self.compute_statistical_sensitivity()

    def build_linear_models(self, persistence=None):
        """ Forces the construction of linear fits per Morse-Smale
            segment.
            @ In, persistence, a floating point value specifying the
            simplification level to use, if this value is None, then we
            will build models based on the internally set persistence
            level for this Morse-Smale object.
        """
        partitions = self.get_partitions(persistence)

        for key, items in partitions.items():
            X = self.Xnorm[np.array(items), :]
            y = np.array(self.Y[np.array(items)])
            w = self.w[np.array(items)]

            (temp_yIntercept, temp_beta_hat) = weighted_linear_model(X, y, w)
            self.segmentFits[key] = np.hstack((temp_yIntercept, temp_beta_hat))

            yHat = X.dot(self.segmentFits[key][1:]) + self.segmentFits[key][0]

            self.segmentFitnesses[key] = sum(np.sqrt((yHat-y)**2))

    def get_label(self, indices=None, applyFilters=False):
        """ Returns the label pair indices requested by the user
            @ In, indices, a list of non-negative integers specifying
            the row indices to return
            @ In, applyFilters, a boolean specifying whether data
            filters should be used to prune the results
            @ Out, a list of integer 2-tuples specifying the minimum and
            maximum index of the specified rows.
        """
        if applyFilters:
            indices = self.get_mask(indices)
        return super(AnnotatedMorseSmaleComplex, self).get_label(indices)

    def get_normed_x(self, rows=None, cols=None, applyFilters=False):
        """ Returns the normalized input data requested by the user
            @ In, rows, a list of non-negative integers specifying the
            row indices to return
            @ In, cols, a list of non-negative integers specifying the
            column indices to return
            @ In, applyFilters, a boolean specifying whether data
            filters should be used to prune the results
            @ Out, a matrix of floating point values specifying the
            normalized data values used in internal computations
            filtered by the three input parameters.
        """
        if applyFilters:
            rows = self.get_mask(rows)

        return super(AnnotatedMorseSmaleComplex, self).get_normed_x(rows, cols)

    def get_selected_extrema(self):
        """
            Returns the extrema highlighted as being selected in an
            attached UI
            @ Out, a list of non-negative integer indices specifying the
            extrema selected.
        """
        return self.selectedExtrema

    def get_selected_segments(self):
        """
            Returns the Morse-Smale segments highlighted as being
            selected in an attached UI
            @ Out, a list of non-negative integer index pairs specifying
            the min-max pairs associated to the selected Morse-Smale
            segments.
        """
        return self.selectedSegments

    def get_weights(self, indices=None, applyFilters=False):
        """ Returns the weights requested by the user
            @ In, indices, a list of non-negative integers specifying
            the row indices to return
            @ In, applyFilters, a boolean specifying whether data
            filters should be used to prune the results
            @ Out, a list of floating point values specifying the
            weights associated to the input data rows filtered by the
            two input parameters.
        """
        if applyFilters:
            indices = self.get_mask(indices)

        return super(AnnotatedMorseSmaleComplex, self).get_weights(indices)

    def get_x(self, rows=None, cols=None, applyFilters=False):
        """ Returns the input data requested by the user
            @ In, rows, a list of non-negative integers specifying the
            row indices to return
            @ In, cols, a list of non-negative integers specifying the
            column indices to return
            @ In, applyFilters, a boolean specifying whether data
            filters should be used to prune the results
            @ Out, a matrix of floating point values specifying the
            input data values filtered by the three input parameters.
        """
        if applyFilters:
            rows = self.get_mask(rows)

        return super(AnnotatedMorseSmaleComplex, self).get_x(rows, cols)

    def get_y(self, indices=None, applyFilters=False):
        """ Returns the output data requested by the user
            @ In, indices, a list of non-negative integers specifying
            the row indices to return
            @ In, applyFilters, a boolean specifying whether data
            filters should be used to prune the results
            @ Out, a list of floating point values specifying the output
            data values filtered by the two input parameters.
        """
        if applyFilters:
            indices = self.get_mask(indices)

        return super(AnnotatedMorseSmaleComplex, self).get_y(indices)

    def predict(self, x, key):
        """ Returns the predicted response of x given a model index
            @ In, x, a list of input values matching the dimensionality
            of the input space
            @ In, key, a 2-tuple specifying a min-max id pair used for
            determining which model is being used for prediction
            @ Out, a predicted response value for the given input point
        """
        beta_hat = self.segmentFits[key][1:]
        y_intercept = self.segmentFits[key][0]
        if len(x.shape) == 1:
            return x.dot(beta_hat) + y_intercept
        else:
            predictions = []
            for xi in x:
                predictions.append(xi.dot(beta_hat) + y_intercept)
            return predictions

    def predict_y(self, indices=None, fit='linear', applyFilters=False):
        """ Returns the predicted output values requested by the user
            @ In, indices, a list of non-negative integers specifying
            the row indices to predict
            @ In, fit, an optional string specifying which fit should be
            used to predict each location, 'linear' = Morse-Smale
            segment, 'maxima' = descending/stable manifold,
            'minima' = ascending/unstable manifold. Only 'linear' is
            available in this version.
            @ In, applyFilters, a boolean specifying whether data
            filters should be used to prune the results
            @ Out, a list of floating point values specifying the
            predicted output values filtered by the three input
            parameters.
        """
        partitions = self.get_partitions(self.persistence)

        predictedY = np.zeros(self.get_sample_size())
        if fit == 'linear':
            for key, items in partitions.items():
                    beta_hat = self.segmentFits[key][1:]
                    y_intercept = self.segmentFits[key][0]
                    for idx in items:
                        predictedY[idx] = self.Xnorm[idx, :].dot(beta_hat) + \
                                          y_intercept
        # Possible extension to fit data per stable or unstable manifold
        # would go here

        if indices is None:
            indices = list(range(0, self.get_sample_size()))
        if applyFilters:
            indices = self.get_mask(indices)
        indices = np.array(sorted(list(set(indices))))
        return predictedY[indices]

    def residuals(self, indices=None, fit='linear', signed=False,
                  applyFilters=False):
        """
            Returns the residual between the output data and the
            predicted output values requested by the user
            @ In, indices, a list of non-negative integers specifying
            the row indices for which to compute residuals
            @ In, fit, an optional string specifying which fit should be
            used to predict each location, 'linear' = Morse-Smale
            segment, 'maxima' = descending/stable manifold,
            'minima' = ascending/unstable manifold
            @ In, applyFilters, a boolean specifying whether data
            filters should be used to prune the results
            @ Out, a list of floating point values specifying the signed
            difference between the predicted output values and the
            original output data filtered by the three input parameters.
        """
        if indices is None:
            indices = list(range(0, self.get_sample_size()))
        else:
            indices = sorted(list(set(indices)))

        if applyFilters:
            indices = self.get_mask(indices)

        indices = np.array(sorted(list(set(indices))))

        yRange = max(self.Y) - min(self.Y)
        actualY = self.get_y(indices)
        predictedY = self.predict_y(indices, fit)
        if signed:
            residuals = (actualY-predictedY)/yRange
        else:
            residuals = np.absolute(actualY-predictedY)/yRange

        return residuals

    def compute_statistical_sensitivity(self):
        """ Computes the per segment Pearson correlation coefficients and the
            Spearman rank correlation coefficients and stores them internally.
        """
        partitions = self.get_partitions()

        self.pearson = {}
        self.spearman = {}
        for key, items in partitions.items():
            X = self.Xnorm[np.array(items), :]
            y = self.Y[np.array(items)]

            self.pearson[key] = []
            self.spearman[key] = []

            for col in range(0, X.shape[1]):
                    self.pearson[key].append(stats.pearsonr(X[:, col], y)[0])
                    self.spearman[key].append(stats.spearmanr(X[:, col], y)[0])
