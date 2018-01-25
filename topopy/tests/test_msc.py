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
"""
    This module will test the basic functionalities of topopy.MorseSmaleComplex
"""
from unittest import TestCase

import numpy as np

import topopy

from .testFunctions import gerber

class TestMSC(TestCase):
    """
    Class for testing the Morse-Smale Complex
    """

    def setup(self):
        """
        Setup function will create a fixed point set and parameter settings for
        testing different aspects of this library.
        """
        pass

    def test_default(self):
        """
        Blank function serving as a template
        """
        self.setup()

        x, y = np.mgrid[0:1:40j, 0:1:40j]
        X = np.vstack([x.ravel(), y.ravel()]).T
        Y = gerber(X)

        msc = topopy.MorseSmaleComplex(debug=True)
        msc.Build(X, Y)

        ########################################################################
        import itertools
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        # color_list = itertools.cycle(['#88CCEE', '#DDCC77', '#AA4499', '#117733', '#332288', '#999933', '#44AA99', '#882255', '#CC6677'])

        color_list = itertools.cycle(['#FF6666', '#FFAA66', '#FFFF66', '#99FF66',
                                    '#66FF66', '#66FFCC', '#66FFFF', '#6699FF',
                                    '#6666FF', '#9966FF', '#FF66FF', '#FF66AA',
                                    '#000000', '#666666', '#999999', '#CCCCCC'])

        fig = plt.figure()        
        edges = []
        for i in range(msc.GetSampleSize()):
            neighbors = msc.GetNeighbors(i)
            for j in neighbors:
                edges.append([i,j])
        np.array(edges)

        lc = LineCollection(X[edges],colors='#CCCCCC')
        plt.gca().add_collection(lc)

        minIdxs = set()
        maxIdxs = set()
        for key,idxs in msc.base_partitions.items():
            minIdx = int(key.split(',')[0])
            maxIdx = int(key.split(',')[1])
            plt.scatter(X[idxs,0],X[idxs,1], c = next(color_list))
            plt.scatter(X[minIdx,0],X[minIdx,1], c = '#006699', marker='v', s=100)
            plt.scatter(X[maxIdx,0],X[maxIdx,1], c = '#990000', marker='^', s=100)
            minIdxs.add(minIdx)
            maxIdxs.add(maxIdx)

        for idx in sorted(list(minIdxs)):
            print('Min: {} ({},{})'.format(idx, X[idx][0], X[idx][1]))

        for idx in sorted(list(maxIdxs)):
            print('Max: {} ({},{})'.format(idx, X[idx][0], X[idx][1]))

        for k in msc.GetNeighbors(999):
            print(k)
        print(msc.GetY([999]+list(msc.GetNeighbors(999))))

        plt.show()
        ########################################################################

        self.assertEqual(len(msc.base_partitions.keys()), 16)
