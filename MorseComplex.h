#ifndef MC_H
#define MC_H

#include <map>
#include <vector>
#include <set>
#include <sstream>
#include <list>
#include <iostream>

/**
 * Merge data structure
 * A integer index to describe the parent and saddle point samples, and
 * a floating point value to describe the persistence of a particular
 * point sample labeled as either a local minimum or a local maximum
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
 * Approximate Morse Complex.
 * Stores the hierarchical decomposition of an arbitrary point cloud
 * according to its estimated gradient flow.
 */
template<typename T>
class MorseComplex
{
 public:
  /* Here are a list of typedefs to make things more compact and
   * readable
   */
  typedef std::pair<int, int> int_pair;

  typedef typename std::map< int_pair, std::pair<T,int> > map_pi_pfi;
  typedef typename map_pi_pfi::iterator map_pi_pfi_it;

  typedef typename std::map< std::pair<T,int>, int_pair> map_pfi_pi;
  typedef typename map_pfi_pi::iterator map_pfi_pi_it;

  typedef typename std::map< int, Merge<T> > persistence_map;
  typedef typename std::map< int, Merge<T> >::iterator persistence_map_it;

  /**
   * Constructor that will decompose a passed in dataset, note the user
   * passes in a list of candidate edges from which it will prune
   * accordingly using ngl
   * @param Xin flattened vector of input data in row-major order
   * @param yin vector of response values in a one-to-one correspondence
   *        with Xin
   * @param gradientMethod string identifier for what type of gradient
   *        estimation method is used
   * @param persistenceType string identifier for what type of
   *        persistence computation should be used
   * @param win vector of probability values in a one-to-one
   *        correspondence with Xin
   * @param neighborhoods a map where the keys are individual indices
   *        and the values are sets of indices that are connected to
   *        that key index
   *        (TODO consider using a vector< set<int> > here)
   */
  MorseComplex(std::vector<T> &Xin, std::vector<T> &yin,
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
   * Returns the global maximum value attained by the output of the
   * associated dataset
   */
  T MaxY();

  /**
   * Returns the global minimum value attained by the output of the
   * associated dataset
   */
  T MinY();

  /**
   * Returns MaxY()-MinY()
   */
  T RangeY();

  /**
   * Returns the maximum value attained by a specified dimension of the
   * input space of the associated dataset
   * @param dim integer specifying the column of data where the
   *        specified input dimension is stored
   */
  T MaxX(int dim);

  /**
   * Returns the minimum value attained by a specified dimension of the
   * input space of the associated dataset
   * @param dim integer specifying the column of data where the
   *        specified input dimension is stored
   */
  T MinX(int dim);

  /**
   * Returns MaxX(dim)-MinX(dim)
   * @param dim integer specifying the column of data where the
   *        specified input dimension is stored
   */
  T RangeX(int dim);

  /**
   * Extracts the input values for a specified sample of the associated
   * data
   * @param i integer specifying the row of data where the specified
   *        sample is stored
   * @param xi a pointer that will be updated to point at the specified
   *        data sample
   */
  void GetX(int i, T *xi);

  /**
   * Extracts the input value for a specified sample and dimension of
   * the associated data
   * @param i integer specifying the row of data where the specified
   *        sample is stored
   * @param j integer specifying the column of data where the specified
   *        input dimension is stored
   */
  T GetX(int i, int j);

  /**
   * Extracts the scalar output value for a specified sample of the
   * associated data
   * @param i integer specifying the row of data where the specified
   *        sample is stored
   */
  T GetY(int i);

  /**
   * Returns the index of the maximum sample to which sample "i" flows
   * to at a specified level of persistence simplification
   * @param i integer specifying the unique sample queried
   * @param pers floating point value specifying an optional amount of
   *        simplification to consider when retrieving the maximum index
   */
  int MaxLabel(int i, T pers);

  /**
   * Returns a formatted string that can be used to determine the merge
   * hierarchy of the topological decomposition of the associated
   * dataset
   */
  std::string to_json();

  /**
   * Returns a sorted list of persistence values for this complex
   **/
   std::vector<T> SortedPersistences();

  /**
   * Returns a map where the key represents a maximum (maxIdx) and the
   * value is a list of associated indices from the input data
   * @param persistence floating point value that optionally simplifies
   *        the topological decomposition before fetching the indices.
   */
  std::map< int, std::vector<int> > GetPartitions(T persistence);

 private:
  /** String identifier specifying how we should compute persistence */
  std::string persistenceType;

  /** Input data matrix */
  std::vector< std::vector<T> > X;
  /** Output data vector */
  std::vector<T> y;
  /** Probability data vector */
  std::vector<T> w;

  /** Maps a list of points that are neighbors of the index */
  std::map< int, std::set<int> > neighbors;

  /**
   * Maps an index pair (smaller vakye first) to the distance between
   * the two points represented by these indices
   */
  std::map< int_pair, T > distances;

  /**
   * Estimated neighbor gradient flow
   */
  std::vector<int> neighborFlow;

  /**
   * Local minimum/maximum index to which each point flows to
   */
  std::vector<int> flow;

  /**
   * The simplification hierarchy for all of the maxima
   * The key is the maximum index and the value is the persistence
   * value, extrema index into which this merges, and the cancelling
   * saddle
   */
  persistence_map maxHierarchy;

  /**
   * Returns the ascending neighbor of the sample specified by index
   * @param index integer specifying the unique sample to query
   */
  int ascending(int index);

  /**
   * Compute and locally store the distances associated to each edge in
   * the graph. This will be used to estimate gradient magnitudes.
   */
  void computeDistances();

  //////////////////////////////////////////////////////////////////////
  // Gradient estimation Methods

  /**
   * Function that will delegate the gradient estimation to the
   * appropriate method
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

  //////////////////////////////////////////////////////////////////////
  //Persistence Simplification

  /**
   * Creates the full merge hierarchy
   */
  void ComputeMaximaPersistence();
};

#endif //MC_H