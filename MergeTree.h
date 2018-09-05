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
   * @param gradientMethod string identifier for what type of gradient
   *        estimation method is used
   * @param neighborhoods a map where the keys are individual indices and the
   *        values are sets of indices that are connected to that key index
   *       (TODO consider using a vector< set<int> > here)
   */
  MergeTree(std::vector<T> &Xin, std::vector<T> &yin,
            std::string gradientMethod,
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
