import sys
import time
import collections
import json

import numpy as np

from .topology import MorseComplexFloat, vectorFloat, mapIntSetInt
from .TopologicalObject import TopologicalObject


class MorseComplex(TopologicalObject):
    """ A wrapper class for the C++ approximate Morse complex Object

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
    simplification : str
        An optional string specifying how we will compute the simplification
        hierarchy. Currently, three modes are supported 'difference',
        'probability' and 'count'. 'difference' will take the function value
        difference of the extrema and its closest function valued neighboring
        saddle (standard persistence simplification), 'probability' will augment
        this value by multiplying the probability of the extremum and its
        saddle, and count will order the simplification by the size (number of
        points) in each manifold such that smaller features will be absorbed
        into neighboring larger features first.
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
        simplification="difference",
        aggregator=None,
        debug=False,
    ):
        super(MorseComplex, self).__init__(
            graph=graph,
            gradient=gradient,
            normalization=normalization,
            aggregator=aggregator,
            debug=debug,
        )
        self.simplification = simplification

    def reset(self):
        """ Empties all internal storage containers


        Returns
        -------
        None

        """
        super(MorseComplex, self).reset()

        self.base_partitions = {}
        self.merge_sequence = {}

        self.persistences = []
        self.max_indices = []

        # State properties
        self.persistence = 0.

    def build(self, X, Y, w=None):
        """ Assigns data to this object and builds the Morse Complex

        Uses an internal graph given in the constructor to build a Morse complex
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
        super(MorseComplex, self).build(X, Y, w)

        if self.debug:
            sys.stdout.write("Decomposition: ")
            start = time.perf_counter()

        edges = mapIntSetInt()
        for key, items in self.graph.full_graph().items():
            items = tuple(items)
            edges[key] = items

        morse_complex = MorseComplexFloat(
            vectorFloat(self.Xnorm.flatten()),
            vectorFloat(self.Y),
            str(self.gradient),
            str(self.simplification),
            vectorFloat(self.w),
            edges,
            self.debug,
        )
        self.__amc = morse_complex

        self.persistences = []
        self.merge_sequence = {}
        morse_complex_json = json.loads(morse_complex.to_json())
        hierarchy = morse_complex_json["Hierarchy"]
        for merge in hierarchy:
            self.persistences.append(merge["Persistence"])
            self.merge_sequence[merge["Dying"]] = (
                merge["Persistence"],
                merge["Surviving"],
                merge["Saddle"],
            )
        self.persistences = sorted(list(set(self.persistences)))

        partitions = morse_complex_json["Partitions"]
        self.base_partitions = {}
        for i, label in enumerate(partitions):
            if label not in self.base_partitions:
                self.base_partitions[label] = []
            self.base_partitions[label].append(i)

        self.max_indices = list(self.base_partitions.keys())

        if self.debug:
            end = time.perf_counter()
            sys.stdout.write("%f s\n" % (end - start))

    def _build_for_morse_smale_complex(self, morse_smale_complex, negate=False):
        Y = morse_smale_complex.Y
        X = morse_smale_complex.Xnorm
        N = len(Y) - 1
        complex_type = "Stable"

        edges = mapIntSetInt()
        for key, items in morse_smale_complex.graph.full_graph().items():
            items = tuple(items)
            edges[key] = items

        if negate:
            Y = -Y[::-1]
            X = X[::-1]
            complex_type = "Unstable"
            reversed_edges = {}
            for key, neighbors in edges.items():
                reversed_edges[N - key] = tuple([N - i for i in neighbors])
            edges = reversed_edges

        if self.debug:
            sys.stdout.write(complex_type + " Decomposition: ")
            start = time.perf_counter()

        morse_complex = MorseComplexFloat(
            vectorFloat(X.flatten()),
            vectorFloat(Y),
            str(morse_smale_complex.gradient),
            str(morse_smale_complex.simplification),
            vectorFloat(morse_smale_complex.w),
            mapIntSetInt(edges),
            morse_smale_complex.debug,
        )

        self.persistences = []
        self.merge_sequence = {}
        morse_complex_json = json.loads(morse_complex.to_json())
        hierarchy = morse_complex_json["Hierarchy"]
        for merge in hierarchy:
            self.persistences.append(merge["Persistence"])
            if negate:
                self.merge_sequence[N - merge["Dying"]] = (
                    merge["Persistence"],
                    N - merge["Surviving"],
                    N - merge["Saddle"],
                )
            else:
                self.merge_sequence[merge["Dying"]] = (
                    merge["Persistence"],
                    merge["Surviving"],
                    merge["Saddle"],
                )
        self.persistences = sorted(list(set(self.persistences)))

        partitions = morse_complex_json["Partitions"]
        self.base_partitions = {}
        for i, label in enumerate(partitions):
            if negate:
                real_label = N - label
                real_index = N - i
            else:
                real_label = label
                real_index = i
            if real_label not in self.base_partitions:
                self.base_partitions[real_label] = []
            self.base_partitions[real_label].append(real_index)

        self.max_indices = list(self.base_partitions.keys())

        if self.debug:
            end = time.perf_counter()
            sys.stdout.write("%f s\n" % (end - start))

    def save(self, filename=None):
        """ Saves a constructed Morse Complex in json file

        Parameters
        ----------
        filename : str
            A filename for storing the hierarchical merging of features and the
            base level partitions of the data

        Returns
        -------
        None

        """
        if filename is None:
            filename = "morse_complex.json"
        with open(filename, "w") as fp:
            fp.write(self.to_json())

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

        Returns
        -------
        dict of int: tuple(float, int, int)
            A dictionary of tuples where the key is the dying extrema and the
            tuple is the the persistence, parent index, and the saddle index
            associated to the dying index, in that order.

        """
        return self.merge_sequence

    def get_partitions(self, persistence=None):
        """ Returns the partitioned data based on a specified persistence level


        Parameters
        ----------
        persistence : float
            A floating point value specifying the size of the smallest feature
            we want to track.
            Default = None means consider all features.

        Returns
        -------
        dict of int: list of int
            A dictionary lists where each key is a integer specifying the index
            of the extremum. Each entry will hold a list of indices specifying
            points that are associated to this extremum.

        """
        if persistence is None:
            persistence = self.persistence
        partitions = {}
        # TODO: Possibly cache at the critical persistence values,
        # previously caching was done at every query level, but that
        # does not make sense as the partitions will only change once
        # the next value in self.persistences is attained. Honestly,
        # this is probably not a necessary optimization that needs to
        # be made. Consider instead, Yarden's way of storing the points
        # such that merged arrays will be adjacent.
        for key, items in self.base_partitions.items():
            new_key = key
            while (
                self.merge_sequence[new_key][0] < persistence
                and self.merge_sequence[new_key][1] != new_key
            ):
                new_key = self.merge_sequence[new_key][1]
            if new_key not in partitions:
                partitions[new_key] = []
            partitions[new_key].extend(items)

        for key in partitions:
            partitions[key] = sorted(list(set(partitions[key])))

        return partitions

    def get_persistence(self):
        """ Retrieves the persistence simplfication level being used for this
        complex

        Returns
        -------
        float
            Floating point value specifying the current persistence setting

        """
        return self.persistence

    def set_persistence(self, p):
        """ Sets the persistence simplfication level to be used for representing
        this complex

        Parameters
        ----------
        p : float
            A floating point value specifying the internally held size of the
            smallest feature we want to track.

        Returns
        -------
        None

        """
        self.persistence = p

    def get_label(self, indices=None):
        """ Returns the label indices requested by the user

        Parameters
        ----------
        indices : list of int
            A list of non-negative integers specifying the row indices to return

        Returns
        -------
        list of int
            A list of integers specifying the extremum index of the specified
            rows.

        """
        if indices is None:
            indices = list(range(0, self.get_sample_size()))
        elif isinstance(indices, collections.Iterable):
            indices = sorted(list(set(indices)))
        else:
            indices = [indices]

        if len(indices) == 0:
            return []
        partitions = self.get_partitions(self.persistence)
        labels = self.X.shape[0] * [None]
        for label, partition_indices in partitions.items():
            for idx in np.intersect1d(partition_indices, indices):
                labels[idx] = label

        labels = np.array(labels)
        if len(indices) == 1:
            return labels[indices][0]
        return labels[indices]

    def get_current_labels(self):
        """ Returns a list of tuples that specifies the extremum index labels
        associated to each input sample

        Returns
        -------
        list of tuple(int, int)
            a list of non-negative integers specifying the extremum-flow indices
            associated to each input sample at the current level of persistence

        """
        partitions = self.get_partitions(self.persistence)
        return partitions.keys()

    def get_sample_size(self, key=None):
        """ Returns the number of samples in the input data

        Parameters
        ----------
        key : int
            An optional integer specifying a max id used for determining which
            partition size should be returned. If not specified then the size of
            the entire data set will be returned.

        Returns
        -------
        int
            An integer specifying the number of samples.

        """
        if key is None:
            return len(self.Y)
        else:
            return len(self.get_partitions(self.persistence)[key])

    def get_classification(self, idx):
        """ Given an index, this function will report whether that sample is a
        local maximum or a regular point.

        Parameters
        ----------
        idx : int
            A non-negative integer less than the sample size of the input data.

        Returns
        -------
        str
            A string specifying the classification type of the input sample:
            will be 'maximum' or 'regular.'

        """
        if idx in self.max_indices:
            return "maximum"
        return "regular"

    def to_json(self):
        """ Writes the complete Morse complex merge hierarchy to a string

        Returns
        -------
        str
            A string storing the entire merge hierarchy of all maxima

        """
        capsule = {}
        capsule["Hierarchy"] = []
        for (
            dying,
            (persistence, surviving, saddle),
        ) in self.merge_sequence.items():
            capsule["Hierarchy"].append(
                {
                    "Persistence": persistence,
                    "Dying": dying,
                    "Surviving": surviving,
                    "Saddle": saddle,
                }
            )
        capsule["Partitions"] = []
        base = np.array([None] * len(self.Y))
        for label, items in self.base_partitions.items():
            base[items] = label
        capsule["Partitions"] = base.tolist()

        return json.dumps(capsule, separators=(",", ":"))
