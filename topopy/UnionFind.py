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


class Singleton(object):
    """ A class for storing a singleton in the Union-Find data structure
    """
    def __init__(self, _idx):
        self.idx = _idx
        self.parent = _idx
        self.rank = 0


class UnionFind(object):
    """
    """
    def __init__(self):
        """
        """
        self.sets = {}

    def make_set(self, idx):
        """
        """
        if idx not in self.sets:
            self.sets[idx] = Singleton(idx)

    def find(self, idx):
        """
        """
        if idx not in self.sets:
            self.make_set(idx)

        if self.sets[idx].parent == idx:
            return idx
        else:
            self.sets[idx].parent = self.find(self.sets[idx].parent)
            return self.sets[idx].parent

    def union(self, x, y):
        """
        """
        xRoot = self.find(x)
        yRoot = self.find(y)
        if xRoot == yRoot:
            return

        if self.sets[xRoot].rank < self.sets[yRoot].rank or \
           (self.sets[xRoot].rank == self.sets[yRoot].rank and xRoot < yRoot):
            self.sets[xRoot].parent = yRoot
            self.sets[yRoot].rank = self.sets[yRoot].rank + 1
        else:
            self.sets[yRoot].parent = xRoot
            self.sets[xRoot].rank = self.sets[xRoot].rank + 1

    def get_components(self):
        """
        """
        collections = {}
        for key in self.sets.keys():
            root = self.find(key)
            if root not in collections:
                    collections[root] = []
            collections[root].append(key)
        return collections.values()
