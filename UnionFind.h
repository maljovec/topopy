#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <vector>

/**
 * A singleton class for the encoding an element of the Union-Find data
 * structure.
 */
class Singleton
{
 public:
  /**
   * Constructor to initialize a Singleton as its own parent and of zero rank.
   * @param _id an integer defining the identifier of this Singleton.
   */
  Singleton(int _id) : id(_id), parent(_id), rank(0) {}
  int id;                                               /** Id of the element */
  int parent;                                  /** Id of the element's parent */
  int rank;                   /** The depth of this element in the Union-Find */
};

/**
 * A Union-Find data structure.
 * Used to store disjoint subsets of Singletons with two main functionalities:
 * finding the subset a Singleton belongs to and joining two disjoint sets.
 */
class UnionFind
{
 public:
  /**
   * Default Constructor acts as an empty initializer.
   */
  UnionFind();

  /**
   * Destructor will release the memory used to store the subsets of Singletons
   * stored by this object.
   */
  ~UnionFind();

  /**
   * A function to create and store a new Singleton as a new subset.
   * @param id an integer defining the new Singleton's identifier, should be
   *        unique.
   */
  void MakeSet(int id);

  /**
   * A function to find a specified identifier's representative by
   * recursively searching the parent nodes until a Singleton is its own parent.
   * @param id an integer identifier for which we want to find a representative.
   */
  int Find(int id);

  /**
   * A function to union two possibly disjoint subsets by first finding their
   * representatives and updating the parent of the lower rank to be the other
   * representative. In case of a tie, the lower identifier will be the parent.
   * This will also increment the rank of the determined parent.
   * @param x an integer identifier of a Singleton we want to merge.
   * @param y an integer identifier of a Singleton we want to merge.
   */
  void Union(int x, int y);

  /**
   * A function to count the number of disjoint sets in the Union-Find
   */
  int CountComponents();

  /**
   * A function to get the representatives of each disjoint set in the
   * Union-Find
   * @param reps a vector of integers that will be populated by this function
   *        with a list of identifiers each representing a disjoint set in the
   *        Union-Find
   */
  void GetComponentRepresentatives(std::vector<int> &reps);

  /**
   * A function to get all of the identifiers associated to a particular
   * representative.
   * @param rep an integer defining the identifier of the representative for
   *        the list of Singleton identifiers. If this value is not actually
   *        a representative of any subset, then the return list will be empty.
   * @param items a vector of integers that will be populated by this function
   *        with a list of identifiers associated to the subset with rep as its
   *        representative.
   */
  void GetComponentItems(int rep, std::vector<int> &items);

 private:
  /**
   * The list of singletons, keyed by their identifiers, use of a map allows for
   * arbitrary indices to be used and does not impose a strict ordering from
   * zero.
   */
  std::map<int, Singleton *> sets;
};

#endif //UNION_FIND_H
