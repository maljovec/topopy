import sys
import time
import collections
import json

import numpy as np

from . import TopologicalObject
from . import MorseComplex


class MorseSmaleComplex(TopologicalObject):
    """ A wrapper class for the C++ approximate Morse-Smale complex
        Object
    """

    def __init__(
        self,
        graph="beta skeleton",
        gradient="steepest",
        max_neighbors=-1,
        beta=1.0,
        normalization=None,
        simplification="difference",
        connect=False,
        aggregator=None,
        debug=False,
    ):
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
        super(MorseSmaleComplex, self).__init__(
            graph=graph,
            gradient=gradient,
            max_neighbors=max_neighbors,
            beta=beta,
            normalization=normalization,
            connect=connect,
            aggregator=aggregator,
            debug=debug,
        )
        self.simplification = simplification

    def reset(self):
        """
            Empties all internal storage containers
        """
        super(MorseSmaleComplex, self).reset()

        self.base_partitions = {}
        self.merge_sequence = {}

        self.persistences = []
        self.min_indices = []
        self.max_indices = []

        # State properties
        self.persistence = 0.

    def build(self, X, Y, w=None, edges=None):
        """ Assigns data to this object and builds the Morse-Smale
            Complex
            @ In, X, an m-by-n array of values specifying m
            n-dimensional samples
            @ In, Y, a m vector of values specifying the output
            responses corresponding to the m samples specified by X
            @ In, w, an optional m vector of values specifying the
            weights associated to each of the m samples used. Default of
            None means all points will be equally weighted
            @ In, edges, an optional list of custom edges to use as a
            starting point for pruning, or in place of a computed graph.
        """
        super(MorseSmaleComplex, self).build(X, Y, w, edges)

        if self.debug:
            sys.stdout.write("Decomposition: ")
            start = time.clock()

        stableManifolds = MorseComplex(debug=self.debug)
        unstableManifolds = MorseComplex(debug=self.debug)

        stableManifolds.build_for_morse_smale_complex(self, False)
        unstableManifolds.build_for_morse_smale_complex(self, True)

        self.min_indices = unstableManifolds.max_indices
        self.max_indices = stableManifolds.max_indices

        # If a degenerate point is both a minimum and a maximum, it
        # could potentially appear twice, but would be masked by the
        # minimum key which would wipe the maximum merge
        self.merge_sequence = stableManifolds.merge_sequence.copy()
        self.merge_sequence.update(unstableManifolds.merge_sequence)
        self.persistences = sorted(
            stableManifolds.persistences + unstableManifolds.persistences
        )

        self.base_partitions = {}
        base = np.array([[None, None]] * len(Y))
        for key, items in unstableManifolds.base_partitions.items():
            base[np.array(items), 0] = key
        for key, items in stableManifolds.base_partitions.items():
            base[np.array(items), 1] = key

        keys = set(map(tuple, base))
        for key in keys:
            self.base_partitions[key] = np.where(
                np.logical_and(base[:, 0] == key[0], base[:, 1] == key[1])
            )[0]

        if self.debug:
            end = time.clock()
            sys.stdout.write("%f s\n" % (end - start))

    def save(self, filename=None):
        """ Saves a constructed Morse-Smale Complex in json file
            @ In, filename, a filename for storing the hierarchical
            merging of features and the base level partitions of the
            data
        """
        if filename is None:
            filename = "morse_smale_complex.json"
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
            @ Out, a dictionary of tuples where the key is the dying
            extrema and the tuple is the the persistence, parent index,
            and the saddle index associated to the dying index, in that
            order.
        """
        return self.merge_sequence

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
            min_index = key[0]
            max_index = key[1]
            while (
                self.merge_sequence[min_index][0] < persistence
                and self.merge_sequence[min_index][1] != min_index
            ):
                min_index = self.merge_sequence[min_index][1]
            while (
                self.merge_sequence[max_index][0] < persistence
                and self.merge_sequence[max_index][1] != max_index
            ):
                max_index = self.merge_sequence[max_index][1]
            new_key = (min_index, max_index)
            if new_key not in partitions:
                partitions[new_key] = []
            partitions[new_key].extend(items.tolist())

        for key in partitions:
            partitions[key] = sorted(list(set(partitions[key])))
        return partitions

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
        partitions = {}
        for key, items in self.base_partitions.items():
            max_index = key[1]
            while (
                self.merge_sequence[max_index][0] < persistence
                and self.merge_sequence[max_index][1] != max_index
            ):
                max_index = self.merge_sequence[max_index][1]
            new_key = max_index
            if new_key not in partitions:
                partitions[new_key] = []
            partitions[new_key].extend(items.tolist())

        for key in partitions:
            partitions[key] = sorted(list(set(partitions[key])))

        return partitions

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
        partitions = {}
        for key, items in self.base_partitions.items():
            min_index = key[0]
            while (
                self.merge_sequence[min_index][0] < persistence
                and self.merge_sequence[min_index][1] != min_index
            ):
                min_index = self.merge_sequence[min_index][1]
            new_key = min_index
            if new_key not in partitions:
                partitions[new_key] = []
            partitions[new_key].extend(items.tolist())

        for key in partitions:
            partitions[key] = sorted(list(set(partitions[key])))

        return partitions

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
            return len(self.get_partitions(self.persistence)[key])

    def get_classification(self, idx):
        """ Given an index, this function will report whether that
            sample is a local minimum, a local maximum, or a regular
            point.
            @ In, idx, a non-negative integer less than the sample size
            of the input data.
            @ Out, a string specifying the classification type of the
            input sample: will be 'maximum,' 'minimum,' or 'regular.'
        """
        if idx in self.min_indices:
            return "minimum"
        elif idx in self.max_indices:
            return "maximum"
        return "regular"

    def to_json(self):
        """ Writes the complete Morse-Smale merge hierarchy to a string
            object.
            @ Out, a string object storing the entire merge hierarchy of
            all minima and maxima.
        """
        capsule = {}
        capsule["Hierarchy"] = []
        for (
            dying,
            (persistence, surviving, saddle),
        ) in self.merge_sequence.items():
            capsule["Hierarchy"].append(
                {
                    "Dying": dying,
                    "Persistence": persistence,
                    "Surviving": surviving,
                    "Saddle": saddle,
                }
            )
        capsule["Partitions"] = []
        base = np.array([None, None] * len(self.Y)).reshape(-1, 2)
        for (min_index, max_index), items in self.base_partitions.items():
            base[items, :] = [min_index, max_index]
        capsule["Partitions"] = base.tolist()

        return json.dumps(capsule)
