/******************************************************************************
 * Software License Agreement (BSD License)                                   *
 *                                                                            *
 * Copyright 2016 University of Utah                                          *
 * Scientific Computing and Imaging Institute                                 *
 * 72 S Central Campus Drive, Room 3750                                       *
 * Salt Lake City, UT 84112                                                   *
 *                                                                            *
 * THE BSD LICENSE                                                            *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * 1. Redistributions of source code must retain the above copyright          *
 *    notice, this list of conditions and the following disclaimer.           *
 * 2. Redistributions in binary form must reproduce the above copyright       *
 *    notice, this list of conditions and the following disclaimer in the     *
 *    documentation and/or other materials provided with the distribution.    *
 * 3. Neither the name of the copyright holder nor the names of its           *
 *    contributors may be used to endorse or promote products derived         *
 *    from this software without specific prior written permission.           *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       *
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    *
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   *
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  *
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      *
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   *
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          *
 ******************************************************************************/

#ifndef MT_H
#define MT_H

#include <map>
#include <vector>
#include <set>
#include <sstream>
#include <list>
#include <iostream>

/**
 * Merge Tree decomposition.
 * Tracks connected components in the sub or super levelsets of a data set.
 */
template<typename T>
class MergeTree
{
 public:
  typedef std::pair<int, int> int_pair;

  /**
   * Constructor that will decompose a passed in dataset, note the user passes
   * in a list of candidate edges from which it will prune accordingly using ngl
   * @param Xin flattened vector of input data in row-major order
   * @param yin vector of response values in a one-to-one correspondence with
   *        Xin
   * @param names a vector of string names for each dimension in the input data
   *        and the name of the response value which should be at the end of the
   *        vector
   * @param gradientMethod string identifier for what type of gradient
   *        estimation method is used
   * @param neighborhoods a map where the keys are individual indices and the
   *        values are sets of indices that are connected to that key index 
   *       (TODO consider using a vector< set<int> > here)
   */
  MergeTree(std::vector<T> &Xin, std::vector<T> &yin,
            std::vector<std::string> &_names, std::string gradientMethod,
            std::map< int, std::set<int> > &neighborhoods,
            bool verbosity=false);

  /**
   * Returns the number of input dimensions in the associated dataset
   */
  int Dimension();

  /**
   * Returns a list of indices marked as neighbors to the specified sample given
   * given by "index"
   * @param index integer specifying the unique sample queried
   */
  std::set<int> Neighbors(int index);


  /**
   * Returns a map where the key represents a critical point in the data and
   * the value is its function value
   */
  std::map<int, T> Nodes();

  /**
   * Returns a map where the key represents a critical point in the data and
   * the value is its function value
   */
  int Root();

  /**
   * Returns a set defining pairs of node keys from an associated call to
   * Nodes() that together specify the merge tree
   */
  std::set<int_pair> Edges();

  /**
   * Returns a map where the keys are each of the nodes in the merge tree and
   * the values are sets of indices that are associated to that component
   * before the next merge
   */
  std::map< int_pair, std::vector<int> > AugmentedEdges();

  /**
   * Returns the number of sample points in the associated dataset
   */
  int Size();

 private:

  std::vector< std::vector<T> > X;                      /** Input data matrix */
  std::vector<T> y;                                    /** Output data vector */
  std::vector<T> w;                               /** Probability data vector */

  std::vector<std::string> names;    /** Names of the input/output dimensions */

  std::map< int, std::set<int> > neighbors;         /** Maps a list of points
                                                     *  that are neighbors of
                                                     *  the index             */

  int root;                               /** The root node index of the tree */

  std::map<int, T> nodes;                     /** The nodes on the merge tree */
  std::set<int_pair> edges;      /** The edges connecting the specified nodes */

  std::map< int_pair, std::vector<int> > augmentedEdges; /** Maps how the
                                                          *  regular vertices
                                                          *  map to arcs. Each
                                                          *  arc will have a
                                                          *  set of indices
                                                          *  that lie on it.  */

};

#endif //MT_H
