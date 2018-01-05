/******************************************************************************
 * Software License Agreement (BSD License)                                   *
 *                                                                            *
 * Copyright 2014 University of Utah                                          *
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

#ifndef AMSC_H
#define AMSC_H

#include <map>
#include <vector>
#include <set>
#include <sstream>
#include <list>
#include <iostream>

/**
 * Merge data structure
 * A integer index to describe the parent and saddle point samples, and a
 * floating point value to describe the persistence of a particular point sample
 * labeled as either a local minimum or a local maximum
 */
template<typename T>
struct Merge
{
  Merge() : persistence(-1),saddle(-1),parent(-1) { }

  Merge(T pers, int saddleIdx, int parentIdx)
    : persistence(pers),saddle(saddleIdx),parent(parentIdx) { }

  T persistence;
  int saddle;
  int parent;
};

/**
 * Discrete gradient flow estimation data structure
 * A pair of integer indices describing upward and downward gradient flow from a
 * point sample.
 */
struct FlowPair
{
  FlowPair(): down(-1), up(-1) { }
  FlowPair(int _down, int _up): down(_down), up(_up) { }

  int down;
  int up;
};

/**
 * Approximate Morse-Smale Complex.
 * Stores the hierarchical decomposition of an arbitrary point cloud according
 * to its estimated gradient flow.
 */
template<typename T>
class AMSC
{
 public:
  /* Here are a list of typedefs to make things more compact and readable */
  typedef std::pair<int, int> int_pair;

  typedef typename std::map< int_pair, std::pair<T,int> > map_pi_pfi;
  typedef typename map_pi_pfi::iterator map_pi_pfi_it;

  typedef typename std::map< std::pair<T,int>, int_pair> map_pfi_pi;
  typedef typename map_pfi_pi::iterator map_pfi_pi_it;

  typedef typename std::map< int, Merge<T> > persistence_map;
  typedef typename std::map< int, Merge<T> >::iterator persistence_map_it;

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
   * @param persistenceType string identifier for what type of persistence
   *        computation should be used
   * @param win vector of probability values in a one-to-one correspondence with
   *        Xin
   * @param neighborhoods a map where the keys are individual indices and the
   *        values are sets of indices that are connected to that key index 
   *       (TODO consider using a vector< set<int> > here)
   */
  AMSC(std::vector<T> &Xin, std::vector<T> &yin,
       std::vector<std::string> &_names, 
       std::string gradientMethod, std::string persistenceType,
       std::vector<T> &win,
       std::map< int, std::set<int> > &neighborhoods, bool verbosity=false);

  /**
   * Returns the number of input dimensions in the associated dataset
   */
  int Dimension();

  /**
   * Returns the number of sample points in the associated dataset
   */
  int Size();

  /**
   * Returns the global maximum value attained by the output of the associated
   * dataset
   */
  T MaxY();

  /**
   * Returns the global minimum value attained by the output of the associated
   * dataset
   */
  T MinY();

  /**
   * Returns MaxY()-MinY()
   */
  T RangeY();

  /**
   * Returns the maximum value attained by a specified dimension of the input
   * space of the associated dataset
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T MaxX(int dim);

  /**
   * Returns the minimum value attained by a specified dimension of the input
   * space of the associated dataset
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T MinX(int dim);

  /**
   * Returns MaxX(dim)-MinX(dim)
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T RangeX(int dim);

  /**
   * Extracts the input values for a specified sample of the associated data
   * @param i integer specifying the row of data where the specified sample
   *        is stored
   * @param xi a pointer that will be updated to point at the specified data
   *        sample
   */
  void GetX(int i, T *xi);

  /**
   * Extracts the input value for a specified sample and dimension of the
   * associated data
   * @param i integer specifying the row of data where the specified sample
   *        is stored
   * @param j integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T GetX(int i, int j);

  /**
   * Extracts the scalar output value for a specified sample of the associated
   * data
   * @param i integer specifying the row of data where the specified sample
   *        is stored
   */
  T GetY(int i);

  /**
   * Returns the index of the minimum sample to which sample "i" flows to at a
   * specified level of persistence simplification
   * @param i integer specifying the unique sample queried
   * @param pers floating point value specifying an optional amount of
   *        simplification to consider when retrieving the minimum index
   */
  int MinLabel(int i, T pers);

  /**
   * Returns the index of the maximum sample to which sample "i" flows to at a
   * specified level of persistence simplification
   * @param i integer specifying the unique sample queried
   * @param pers floating point value specifying an optional amount of
   *        simplification to consider when retrieving the maximum index
   */
  int MaxLabel(int i, T pers);

  /**
   * Returns the string name associated to the specified dimension index
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  std::string Name(int dim);

  /**
   * Returns a list of indices marked as neighbors to the specified sample given
   * given by "index"
   * @param index integer specifying the unique sample queried
   */
  std::set<int> Neighbors(int index);

  /**
   * Returns a formatted string that can be used to determine the merge
   * hierarchy of the topological decomposition of the associated dataset
   */
  std::string PrintHierarchy();

  /**
   * Returns a sorted list of persistence values for this complex
   **/
   std::vector<T> SortedPersistences();

  /**
   * Returns an xml-formatted string that can be used to determine the merge
   * hierarchy of the topological decomposition of the associated dataset
   */
  std::string XMLFormattedHierarchy();

  /**
   * Returns a map where the key represent a minimum/maximum pair
   * ('minIdx,maxIdx') and the value is a list of associated indices from the
   * input data
   * @param persistence floating point value that optionally simplifies the
   *        topological decomposition before fetching the indices.
   */
  std::map< std::string, std::vector<int> > GetPartitions(T persistence);

  /**
   * Returns a map where the key represent a maximum and the value is a list of
   * associated indices from the input data
   * @param persistence floating point value that optionally simplifies the
   *        topological decomposition before fetching the indices.
   */
  std::map< int, std::vector<int> > GetStableManifolds(T persistence);

  /**
   * Returns a map where the key represent a minimum and the value is a list of
   * associated indices from the input data
   * @param persistence floating point value that optionally simplifies the
   *        topological decomposition before fetching the indices.
   */
  std::map< int, std::vector<int> > GetUnstableManifolds(T persistence);

//  std::string ComputeLinearRegressions(T persistence);

 private:
  std::string persistenceType;          /** A string identifier specifying    *
                                         *  how we should compute persistence */

  std::vector< std::vector<T> > X;                      /** Input data matrix */
  std::vector<T> y;                                    /** Output data vector */
  std::vector<T> w;                               /** Probability data vector */

  std::vector<std::string> names;    /** Names of the input/output dimensions */

  std::map< int, std::set<int> > neighbors;         /** Maps a list of points *
                                                     *  that are neighbors of *
                                                     *  the index             */

  std::map< int_pair, T > distances;                /** Maps an index pair    *
                                                     * (smaller vakye first)  *
                                                     * to the distance        *
                                                     * between the two points *
                                                     * represented by these   *
                                                     * indices                */

  std::vector<FlowPair> neighborFlow;         /** Estimated neighbor gradient
                                               * flow first=desc,second = asc */

  std::vector<FlowPair> flow;               /** Local minimum/maximum index to
                                             *  which each point flows from/to
                                             *  first = min, second = max     */

  int globalMinIdx;         /** The index of the overall global minimum point */
  int globalMaxIdx;         /** The index of the overall global maximum point */

  //////////////////////////////////////////////////////////////////////////////
  // Key is my index and the value is the persistence value, extrema index that
  // I merge to, and which saddle I go through
  persistence_map maxHierarchy;       /** The simplification hierarchy for all
                                        * of the maxima                       */

  persistence_map minHierarchy;       /** The simplification hierarchy for all
                                        * of the minima                       */
  //////////////////////////////////////////////////////////////////////////////

  // Private Methods

  /**
   * Returns the ascending neighbor of the sample specified by index
   * @param index integer specifying the unique sample to query
   */
  int ascending(int index);

  /**
   * Returns the descending neighbor of the sample specified by index
   * @param index integer specifying the unique sample to query
   */
  int descending(int index);

  /**
   * Compute and locally store the distances associated to each edge in the
   * graph. This will be used to estimate gradient magnitudes.
   */
  void computeDistances();

  // Gradient estimation Methods

  /**
   * Function that will delegate the gradient estimation to the appropriate
   * method
   * @param method
   */
  void EstimateIntegralLines(std::string method);

  /**
   * Implements the Max Flow algorithm (TODO)
   */
  void MaxFlow();

  /**
   * Implements the Steepest Edge algorithm
   */
  void SteepestEdge();

  //Persistence Simplification

  /**
   * Implements the Steepest Edge algorithm
   */
  void ComputeMaximaPersistence();

  /**
   * Implements the Steepest Edge algorithm
   */
  void ComputeMinimaPersistence();
};

#endif //AMSC_H
