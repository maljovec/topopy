#include "MergeTree.h"
#include "UnionFind.h"
#include "utils.h"

#include <algorithm>
#include <utility>
#include <limits>
#include <sstream>
#include <cstdlib>
#include <time.h>
#include <cstring>

template <class V>
class vector_less
{
private:
  typedef typename V::size_type size_type;
  typedef typename V::value_type value_type;
  vector_less();

  const V& data;
public:
  vector_less(const V& vec) : data(vec) { }
  bool operator() (const size_type& left, const size_type& right) const
  {
    return std::less<value_type> () ( data[left], data[right] );
  }
};

template<typename T>
MergeTree<T>::MergeTree(std::vector<T> &Xin,
                        std::vector<T> &yin,
                        std::string gradientMethod,
                        std::map< int, std::set<int> > &neighborhoods,
                        bool verbosity)
{
  globalVerbosity = verbosity;
  time_t myTime;

  DebugTimerStart(myTime, "\rInitializing...");

  int M = Xin.size() / yin.size();
  int N = yin.size();

  X = std::vector< std::vector<T> >(M, std::vector<T>(N, 0));
  for(int n = 0; n < N; n++)
  {
    for(int m = 0; m < M; m++)
      X[m][n] = Xin[n*M+m];
  }
  y = yin;
  w = std::vector<T>(y.size(), 1);

  neighbors = neighborhoods;

  DebugTimerStop(myTime);

  DebugTimerStart(myTime, "\rSorting data...");

  std::vector<int> sortedIndices(N);
  for ( int i = 0; i < N; i++)
    sortedIndices[i] = i;

  // Clever, but does not work for some reason:
  // auto comparator = [&yin](int a, int b){ return yin[a] < yin[b]; };
  // std::stable_sort(sortedIndices.begin(), sortedIndices.end(), comparator);
  std::stable_sort(sortedIndices.begin(), sortedIndices.end(), vector_less< std::vector<T> >(y));

  DebugTimerStop(myTime);

  DebugTimerStart(myTime, "\rCreating Singletons...");
  UnionFind components;
  for (int i = 0; i < N; i++)
  {
    components.MakeSet(i);
  }
  DebugTimerStop(myTime);

  DebugTimerStart(myTime, "\rWalking data...");

  // This will map the arbitrary representative used by the union-find to the
  // actual next node to be merged from that connected component
  std::map<int,int> rep_to_node;

  std::map< int, std::vector<int> > augmentedEdgesToLowerNode;

  // Walk the sorted vertices
  for (int i = 0; i < N; i++)
  {
    int idx = sortedIndices[i];
    std::set<int> lowerComponents;

    // Look at each neighbor
    for (std::set<int>::iterator it = neighbors[idx].begin();
         it != neighbors[idx].end(); it++)
    {
      int k = *it;
      if (y[k] < y[idx])
      {
        lowerComponents.insert(components.Find(k));
      }
    }
    // A new connected component is born
    if (lowerComponents.size() == 0)
    {
      nodes[idx] = y[idx];

      //Update this new component to see the minimum as its representative
      rep_to_node[components.Find(idx)] = idx;
    }
    // Two or more connected components are merged
    else if (lowerComponents.size() > 1)
    {
      nodes[idx] = y[idx];

      for (std::set<int>::iterator it = lowerComponents.begin();
        it != lowerComponents.end(); it++ )
      {
        int rep = *it;
        int idx2 = rep_to_node[rep];

        std::pair<int, int> newEdge = std::make_pair(idx2,idx);
        edges.insert(newEdge);
        if (augmentedEdgesToLowerNode.find(idx2) != augmentedEdgesToLowerNode.end())
        {
          augmentedEdges[newEdge] = augmentedEdgesToLowerNode[idx2];
        }

        // Merge
        components.Union(rep,idx);

        // Update the new connected component to point to the saddle
        rep = components.Find(rep);
        rep_to_node[rep] = idx;
      }
    }
    // This point is regular, merge it with its lower component
    else
    {
      int rep = *(lowerComponents.begin());
      components.Union(rep,idx);
      int idx2 = rep_to_node[rep];
      augmentedEdgesToLowerNode[idx2].push_back(idx);
    }
  }
  DebugTimerStop(myTime);

  root = sortedIndices[N-1];
  int treeTopIdx = rep_to_node[components.Find(root)];
  nodes[root] = y[root];

  std::pair<int, int> newEdge = std::make_pair(treeTopIdx,root);
  edges.insert(std::make_pair(treeTopIdx,root));
  if (augmentedEdgesToLowerNode.find(treeTopIdx) != augmentedEdgesToLowerNode.end())
  {
    augmentedEdges[newEdge] = augmentedEdgesToLowerNode[treeTopIdx];
  }

  // Always reset the global verbosity to false
  globalVerbosity = false;
}

template<typename T>
int MergeTree<T>::Dimension()
{
  return (int) X.size();
}

template<typename T>
std::set<int> MergeTree<T>::Neighbors(int index)
{
  return neighbors[index];
}

template<typename T>
int MergeTree<T>::Size()
{
  return y.size();
}

template<typename T>
std::map<int, T> MergeTree<T>::Nodes()
{
  return nodes;
}

template<typename T>
int MergeTree<T>::Root()
{
  return root;
}

template<typename T>
std::set< std::pair<int, int> > MergeTree<T>::Edges()
{
  return edges;
}

template<typename T>
std::map< std::pair<int, int>, std::vector<int> > MergeTree<T>::AugmentedEdges()
{
  return augmentedEdges;
}

template class MergeTree<double>;
template class MergeTree<float>;

