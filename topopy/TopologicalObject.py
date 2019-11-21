import sys
import time
import warnings

import numpy as np
import sklearn.preprocessing

import nglpy as ngl
# import nglpy_cuda as ngl


class TopologicalObject(object):
    """ A base class for housing common interactions between Morse and
        Morse-Smale complexes, and Contour and Merge Trees

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
    short_circuit : bool
        An optional boolean flag for whether the contour tree should be short
        circuited. Enabling this will speed up the processing by bypassing the
        fully augmented search and only focusing on partially augmented split
        and join trees

    """

    precision = 16

    @staticmethod
    def aggregate_duplicates(X, Y, aggregator="mean", precision=precision):
        """ A function that will attempt to collapse duplicates in domain
            space, X, by aggregating values over the range space, Y.

        Parameters
        ----------
        X : np.ndarray
            An m-by-n array of values specifying m n-dimensional samples
        Y : np.array
            A m vector of values specifying the output responses corresponding
            to the m samples specified by X
        aggregator : str
            An optional string or callable object that specifies what type of
            aggregation to do when duplicates are found in the domain space.
            Default value is mean meaning the code will calculate the mean range
            value over each of the unique, duplicated samples.
        precision : int
            An optional positive integer specifying how many digits numbers
            should be rounded to in order to determine if they are unique or
            not.

        Returns
        -------
        tuple(np.ndarray, np.array)
            A tuple where the first value is an m'-by-n array specifying the
            unique domain samples and the second value is an m' vector
            specifying the associated range values. m' <= m.

        """
        if callable(aggregator):
            pass
        elif "min" in aggregator.lower():
            aggregator = np.min
        elif "max" in aggregator.lower():
            aggregator = np.max
        elif "median" in aggregator.lower():
            aggregator = np.median
        elif aggregator.lower() in ["average", "mean"]:
            aggregator = np.mean
        elif "first" in aggregator.lower():

            def aggregator(x):
                return x[0]

        elif "last" in aggregator.lower():

            def aggregator(x):
                return x[-1]

        else:
            warnings.warn(
                'Aggregator "{}" not understood. Skipping sample '
                "aggregation.".format(aggregator)
            )
            return X, Y

        is_y_multivariate = Y.ndim > 1

        X_rounded = X.round(decimals=precision)
        unique_xs = np.unique(X_rounded, axis=0)

        old_size = len(X_rounded)
        new_size = len(unique_xs)
        if old_size == new_size:
            return X, Y

        if not is_y_multivariate:
            Y = np.atleast_2d(Y).T

        reduced_y = np.empty((new_size, Y.shape[1]))

        warnings.warn(
            "Domain space duplicates caused a data reduction. "
            + "Original size: {} vs. New size: {}".format(old_size, new_size)
        )
        for col in range(Y.shape[1]):
            for i, distinct_row in enumerate(unique_xs):
                filtered_rows = np.all(X_rounded == distinct_row, axis=1)
                reduced_y[i, col] = aggregator(Y[filtered_rows, col])

        if not is_y_multivariate:
            reduced_y = reduced_y.flatten()

        return unique_xs, reduced_y

    def __init__(
        self,
        graph=None,
        gradient="steepest",
        normalization=None,
        aggregator=None,
        debug=False,
    ):
        super(TopologicalObject, self).__init__()
        self.reset()

        if graph is None:
            graph = ngl.EmptyRegionGraph()
        self.graph = graph
        self.gradient = gradient
        self.normalization = normalization
        self.debug = debug
        self.aggregator = aggregator

    def reset(self):
        """ Empties all internal storage containers


        Returns
        -------
        None

        """
        self.X = []
        self.Y = []
        self.w = []

        self.Xnorm = []

    def __set_data(self, X, Y, w=None):
        """ Internally assigns the input data and normalizes it
            according to the user's specifications
            @ In, X, an m-by-n array of values specifying m
            n-dimensional samples
            @ In, Y, a m vector of values specifying the output
            responses corresponding to the m samples specified by X
            @ In, w, an optional m vector of values specifying the
            weights associated to each of the m samples used. Default of
            None means all points will be equally weighted
        """
        self.X = X
        self.Y = Y
        self.check_duplicates()

        if w is not None:
            self.w = np.array(w)
        else:
            self.w = np.ones(len(Y)) * 1.0 / float(len(Y))

        if self.normalization == "feature":
            # This doesn't work with one-dimensional arrays on older
            # versions of sklearn
            min_max_scaler = sklearn.preprocessing.MinMaxScaler()
            self.Xnorm = min_max_scaler.fit_transform(np.atleast_2d(self.X))
        elif self.normalization == "zscore":
            self.Xnorm = sklearn.preprocessing.scale(
                self.X, axis=0, with_mean=True, with_std=True, copy=True
            )
        else:
            self.Xnorm = np.array(self.X)

    def build(self, X, Y, w=None):
        """ Assigns data to this object and builds the requested topological
            structure

        Uses an internal graph given in the constructor to build a topological
        object on the passed in data. Weights are currently ignored.

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
        self.reset()

        if X is None or Y is None:
            return

        self.__set_data(X, Y, w)

        if self.debug:
            sys.stdout.write("Graph Preparation: ")
            start = time.perf_counter()

        self.graph.build(self.Xnorm)

        if self.debug:
            end = time.perf_counter()
            sys.stdout.write("%f s\n" % (end - start))

    def load_data_and_build(self, filename, delimiter=","):
        """ Convenience function for directly working with a data file.

        This opens a file and reads the data into an array, sets the data as an
        nparray and list of dimnames

        Parameters
        ----------
        filename : str
            string representing the data file

        Returns
        -------
        None

        """
        data = np.genfromtxt(
            filename, dtype=float, delimiter=delimiter, names=True
        )
        data = data.view(np.float64).reshape(data.shape + (-1,))

        X = data[:, 0:-1]
        Y = data[:, -1]

        self.build(X=X, Y=Y)

    def get_normed_x(self, rows=None, cols=None):
        """ Returns the normalized input data requested by the user.


        Parameters
        ----------
        rows : list of int
            A list of non-negative integers specifying the row indices to return
        cols : list of int
            A list of non-negative integers specifying the column indices to
            return

        Returns
        -------
        np.ndarray
            A matrix of floating point values specifying the normalized data
            values used in internal computations filtered by the three input
            parameters.

        """
        if rows is None:
            rows = list(range(0, self.get_sample_size()))
        if cols is None:
            cols = list(range(0, self.get_dimensionality()))

        if not hasattr(rows, "__iter__"):
            rows = [rows]
        rows = sorted(list(set(rows)))

        retValue = self.Xnorm[rows, :]
        return retValue[:, cols]

    def get_x(self, rows=None, cols=None):
        """ Returns the input data requested by the user


        Parameters
        ----------
        rows : list of int
            A list of non-negative integers specifying the row indices to return
        cols : list of int
            A list of non-negative integers specifying the column indices to
            return

        Returns
        -------
        np.ndarray
            A matrix of floating point values specifying the input data values
            filtered by the two input parameters.

        """
        if rows is None:
            rows = list(range(0, self.get_sample_size()))
        if cols is None:
            cols = list(range(0, self.get_dimensionality()))

        if not hasattr(rows, "__iter__"):
            rows = [rows]
        rows = sorted(list(set(rows)))

        retValue = self.X[rows, :]
        if len(rows) == 0:
            return []
        return retValue[:, cols]

    def get_y(self, indices=None):
        """ Returns the output data requested by the user

        Parameters
        ----------
        indices : list of int
            A list of non-negative integers specifying the row indices to return

        Returns
        -------
        np.array
            An array of floating point values specifying the output data values
            filtered by the indices input parameter.

        """
        if indices is None:
            indices = list(range(0, self.get_sample_size()))
        else:
            if not hasattr(indices, "__iter__"):
                indices = [indices]
            indices = sorted(list(set(indices)))

        if len(indices) == 0:
            return []
        return self.Y[indices]

    def get_weights(self, indices=None):
        """ Returns the weights requested by the user

        Parameters
        ----------
        indices : list of int
            A list of non-negative integers specifying the row indices to return

        Returns
        -------
        np.array
            An array of floating point values specifying the weights associated
            to the input data rows filtered by the indices input parameter.

        """
        if indices is None:
            indices = list(range(0, self.get_sample_size()))
        else:
            indices = sorted(list(set(indices)))

        if len(indices) == 0:
            return []
        return self.w[indices]

    def get_sample_size(self):
        """ Returns the number of samples in the input data


        Returns
        -------
        int
            Integer specifying the number of samples.

        """
        return len(self.Y)

    def get_dimensionality(self):
        """ Returns the dimensionality of the input space of the input data


        Returns
        -------
        int
            Integer  specifying the dimensionality of the input samples.

        """
        return self.X.shape[1]

    def get_neighbors(self, idx):
        """ Returns a list of neighbors for the specified index


        Parameters
        ----------
        idx : int
            An integer specifying the query point

        Returns
        -------
        list of int
            Integer list of neighbors indices

        """
        return self.graph.neighbors(int(idx))

    def check_duplicates(self):
        """ Function to test whether duplicates exist in the input or output
        space.

        First, if an aggregator function has been specified, the domain space
        duplicates will be consolidated using the function to generate a new
        range value for that shared point. Otherwise, it will raise a
        ValueError. The function will raise a warning if duplicates exist in the
        output space


        Returns
        -------
        None

        """

        if self.aggregator is not None:
            X, Y = TopologicalObject.aggregate_duplicates(
                self.X, self.Y, self.aggregator
            )
            self.X = X
            self.Y = Y

        temp_x = self.X.round(decimals=TopologicalObject.precision)
        unique_xs = len(np.unique(temp_x, axis=0))

        # unique_ys = len(np.unique(self.Y, axis=0))
        # if len(self.Y) != unique_ys:
        #     warnings.warn('Range space has duplicates. Simulation of '
        #                   'simplicity may help, but artificial noise may '
        #                   'occur in flat regions of the domain. Sample size:'
        #                   '{} vs. Unique Records: {}'.format(len(self.Y),
        #                                                      unique_ys))

        if len(self.X) != unique_xs:
            raise ValueError(
                "Domain space has duplicates. Try using an "
                "aggregator function to consolidate duplicates "
                "into a single sample with one range value. "
                "e.g., " + self.__class__.__name__ + "(aggregator='max'). "
                "\n\tNumber of "
                "Records: {}\n\tNumber of Unique Records: {}\n".format(
                    len(self.X), unique_xs
                )
            )
