#include "MorseComplex.h"
#include "UnionFind.h"
#include "utils.h"

#include <algorithm>
#include <utility>
#include <limits>
#include <sstream>
#include <cstdlib>
#include <time.h>
#include <cstring>
#include <cmath>

int followChain(int i, std::map<int,int> merge)
{
  while(merge[i] != i)
    i = merge[i];
  return i;
}

template<typename T>
int MorseComplex<T>::ascending(int index)
{
  return neighborFlow[index];
}

template<typename T>
void MorseComplex<T>::SteepestEdge()
{
  for( int i = 0; i < Size(); i++)
    neighborFlow.push_back(-1);

  //Store the gradient magnitude of each point's largest ascent/descent
  // so we can verify if the next neighbor represents a larger jump.
  std::vector<T> G = std::vector<T>(Size(), 0);

  //compute steepest ascending neighbors
  for(int i=0; i < (int)Size(); i++)
  {
    std::set<int> Ni = neighbors[i];
    int j; // An index adjacent to i

    for (std::set<int>::iterator it = Ni.begin(); it != Ni.end(); it++)
    {
      j = *it;

      double g = (y[j] - y[i]) / distances[sortedPair(i,j)];
      //Compare to i's neighborhood
      if(G[i] < g)
      {
        //j is a steeper ascent than current
        G[i] = g;
        neighborFlow[i] = j;
      }
      else if(G[i] == g
           && neighborFlow[i] != -1
           && neighborFlow[i] < j)
      {
        //j is as steep an ascent as current,
        // and j is a larger index than current
        G[i] = g;
        neighborFlow[i] = j;
      }
      else if(G[i] == g
          && neighborFlow[i] == -1
          && i < j)
      {
        //j is as steep an ascent as current,
        // and current is not set and j is larger than i
        G[i] = g;
        neighborFlow[i] = j;
      }

      //Look to j's neighborhood
      if(G[j] < -g)
      {
        //i is a steeper ascent than current
        G[j] = -g;
        neighborFlow[j] = i;
      }
      else if(G[j] == -g
           && neighborFlow[j] != -1
           && neighborFlow[j] < i)
      {
        //i is as steep an ascent as current,
        // and i is a larger index than current
        G[j] = -g;
        neighborFlow[j] = i;
      }
      else if(G[j] == -g
           && neighborFlow[j] == -1
           && j < i)
      {
        //i is as steep an ascent as current,
        // and current is not set and i is larger than j
        G[j] = -g;
        neighborFlow[j] = i;
      }
    }
  }

  //compute for each point its maximum based on steepest ascent
  for(int i = 0; i < Size(); i++)
    flow.push_back(-1);

  std::list<int> path;
  for(int i=0; i < Size(); i++)
  {
    //If we have not identified this point's maximum, then we will do so
    // now
    if( flow[i] == -1)
    {
      //Recursively trace the upward flow from this point along path,
      // until we reach a point that has no upward flow
      path.clear();
      int prev = i;
      while(prev != -1 && flow[prev] == -1)
      {
        path.push_back(prev);
        prev = ascending(prev);
      }
      int ext = -1;
      if(prev == -1)
      {
        ext = path.back();
        if(this->persistenceType.compare("difference") == 0)
        {
          maxHierarchy[ext] = Merge<T>(RangeY(),ext,ext);
        }
        else if(this->persistenceType.compare("count") == 0)
        {
          maxHierarchy[ext] = Merge<T>(Size(),ext,ext);
        }
        else if(this->persistenceType.compare("probability") == 0)
        {
          maxHierarchy[ext] = Merge<T>(1,ext,ext);
        }
      }
      else
        ext = flow[prev];

      for(std::list<int>::iterator it = path.begin(); it!=path.end(); ++it)
        flow[*it] = ext;
    }
  }
}

//TODO: figure out how to do the probabilistic trace in a Markov chain.
template<typename T>
void MorseComplex<T>::MaxFlow()
{
  for( int i = 0; i < Size(); i++)
    neighborFlow.push_back(-1);

  T *avgGradient = new T[Dimension()];
  T *neighborGradient = new T[Dimension()];
  //compute steepest asc/descending neighbors
  for(int i=0; i < Size(); i++)
  {
    int actualNeighborCount = 0;
    for(int d = 0; d < Dimension(); d++)
      avgGradient[d] = 0;

    std::set<int> Ni = neighbors[i];
    int j; // An index adjacent to i

    for (std::set<int>::iterator it = Ni.begin(); it != Ni.end(); it++)
    {
      j = *it;

      T deltaY = (y[j] - y[i]);
      for(int d = 0; d < Dimension(); d++)
        avgGradient[d] += deltaY / ((X[d][j] - X[d][i]) / distances[sortedPair(i,j)]);
      actualNeighborCount++;
    }

    T *probability = new T[actualNeighborCount];
    for(int d = 0; d < Dimension(); d++)
      avgGradient[d] /= actualNeighborCount;

    actualNeighborCount = 0;
    T probabilitySum = 0;

    for (std::set<int>::iterator it = Ni.begin(); it != Ni.end(); it++)
    {
      j = *it;

      T dot = 0;
      for(int d = 0; d < Dimension(); d++)
      {
        dot += avgGradient[d]*((X[d][j] - X[d][i]));
      }
      dot = dot < 0 ? 0 : dot;
      probability[actualNeighborCount] = dot;
      probabilitySum += dot;
      actualNeighborCount++;
    }

    T randomNumber = rand()/(T) RAND_MAX;

    actualNeighborCount = 0;
    T runningTotal = 0;

    for (std::set<int>::iterator it = Ni.begin(); it != Ni.end(); it++)
    {
      j = *it;

      probability[actualNeighborCount] /= probabilitySum;
      runningTotal += probability[actualNeighborCount];
      if(randomNumber < runningTotal)
      {
        neighborFlow[i] = j;
      }
      actualNeighborCount++;
    }
    delete [] probability;
  }

  //Must now compute flow from neighborFlow

  delete [] avgGradient;
  delete [] neighborGradient;
}

template<typename T>
void MorseComplex<T>::EstimateIntegralLines(std::string method)
{
  if( method.compare("steepest") == 0)
    SteepestEdge();
 // else if(method.compare("maxflow") == 0)
 //   MaxFlow();
  else
  {
    //TODO
    //These checks can probably be done upfront, so as not to waste computation
    std::cerr << "Invalid gradient type: " << method << std::endl;
    exit(1);
  }
}

template<typename T>
void MorseComplex<T>::ComputeMaximaPersistence()
{
  //initial persistences
  //store as pairs of extrema such that p.first merges to p.second (e.g.
  //p.second is the max with the larger function value
  map_pi_pfi pinv;
  for(int i = 0; i < Size(); i++)
  {
    int e1 = flow[i];
    int saddleIdx;

    std::set<int> Ni = neighbors[i];
    int j; // An index adjacent to i

    for (std::set<int>::iterator it = Ni.begin(); it != Ni.end(); it++)
    {
      j = *it;

      int e2 = flow[j];
      if(e1 != e2)
      {
        T pers = 0;
        //p should have the merging index in the first position and the parent
        // in the second
        int_pair p(e1,e2);
        if( y[e1] > y[e2] || (y[e1] == y[e2] && e1 > e2) )
        {
          p.first = e2;
          p.second = e1;
        }

        saddleIdx = y[i] < y[j] ? i : j;

        if (this->persistenceType.compare("difference") == 0)
        {
          pers = y[p.first] - y[saddleIdx];
        }
        else if (this->persistenceType.compare("probability") == 0)
        {
          T probabilityIntegral = 0.;
          int count = 0;

          for(int idx = 0; idx < Size(); idx++)
          {
            if (flow[idx] == p.first)
            {
              probabilityIntegral += w[idx];
              count++;
            }
          }
          pers = probabilityIntegral;
          if (count > 0)
            pers /= (T) count;
        }
        else if (this->persistenceType.compare("count") == 0)
        {
          //TODO: test
          int count = 0;
          for(int idx = 0; idx < Size(); idx++)
            if (flow[idx] == p.first)
              count++;
          pers = count;
        }
        else if (this->persistenceType.compare("area") == 0)
        {
          //FIXME: implement this & test
        }

        map_pi_pfi_it it = pinv.find(p);
        if(it!=pinv.end())
        {
          T tmpPers = (*it).second.first;
          int tmpSaddle = (*it).second.second;
          if(pers < tmpPers || (pers == tmpPers && tmpSaddle < saddleIdx))
          {
            (*it).second = std::pair<T,int>(pers,saddleIdx);
            maxHierarchy[p.first].parent = p.second;
            maxHierarchy[p.first].saddle = saddleIdx;
          }
        }
        else
        {
          pinv[p] = std::pair<T,int>(pers,saddleIdx);
          maxHierarchy[p.first].parent = p.second;
          maxHierarchy[p.first].saddle = saddleIdx;
        }
      }
    }
  }

  map_pfi_pi persistence;
  for(map_pi_pfi_it it = pinv.begin(); it != pinv.end(); ++it)
  {
    persistence[(*it).second] = (*it).first;
  }

  //compute final persistences - recursively merge smallest persistence
  //extrema and update remaining peristencies depending on the merge

  //First, set each maximum to merge into itself
  std::map<int,int> merge;
  for(persistence_map_it it = maxHierarchy.begin();
      it != maxHierarchy.end();
      it++)
  {
    merge[it->first] = it->first;
  }

  map_pfi_pi ptmp;
  map_pi_pfi pinv2;
  while(!persistence.empty())
  {
    map_pfi_pi_it it = persistence.begin();
    int_pair p = (*it).second;

    //store old extrema merging pair and persistence
    int_pair pold = p;
    double pers = (*it).first.first;
    int saddleIdx = (*it).first.second;

    //find new marging pair, based on possible previous merges
    //make sure that p.first is the less significant extrema as before
    p.first = followChain(p.first,merge);
    p.second = followChain(p.second,merge);

    if( y[p.first] > y[p.second] )
      std::swap(p.second, p.first);

    //remove current merge pair from list
    persistence.erase(it);

    //are the extrema already merged?
    if(p.first == p.second)
      continue;


    if (this->persistenceType.compare("difference") == 0)
    {
      //check if there is new merge pair with increased persistence (or same
      // persistence and a larger index maximum)
      T diff = y[p.first] - y[pold.first];
      if( diff > 0 || (diff == 0 && p.first > pold.first ))
      {
        //if the persistence increased insert into the persistence list and
        //merge possible other extrema with smaller persistence values first
        double npers = pers + diff;
        persistence[std::pair<T,int>(npers,saddleIdx)] = p;
      }
      //otherwise merge the pair
      else
      {
        //check if the pair has not been previously merged
        map_pi_pfi_it invIt = pinv2.find(p);
        if(pinv2.end() == invIt)
        {
          merge[p.first] = p.second;
          maxHierarchy[p.first].persistence = pers;
          maxHierarchy[p.first].parent = p.second;
          maxHierarchy[p.first].saddle = saddleIdx;

          ptmp[std::pair<T,int>(pers,saddleIdx)] = p;
          pinv2[p] = std::pair<T,int>(pers,saddleIdx);
        }
      }
    }
    else if (this->persistenceType.compare("probability") == 0)
    {
      T newPersistence = 0;
      T oldPersistence = 0;

      int newCount = 0;
      int oldCount = 0;

      for(int idx = 0; idx < Size(); idx++)
      {
        int extIdx = followChain(flow[idx], merge);
        if (extIdx == p.first)
        {
          newPersistence += w[idx];
          newCount++;
        }
        if (extIdx == pold.first)
        {
          oldPersistence += w[idx];
          oldCount++;
        }
      }

      if (newCount > 0)
        newPersistence /= (T)newCount;

      if (oldCount > 0)
        oldPersistence /= (T)oldCount;

      //check if there is new merge pair with increased persistence (or same
      // persistence and a larger index maximum)
      T diff = newPersistence - oldPersistence;
      if( diff > 0 || (diff == 0 && p.first > pold.first ))
      {
        //if the persistence increased insert into the persistence list and
        //merge possible other extrema with smaller persistence values first
        double npers = newPersistence;
        persistence[std::pair<T,int>(npers,saddleIdx)] = p;
      }
      //otherwise merge the pair
      else
      {
        //check if the pair has not been previously merged
        map_pi_pfi_it invIt = pinv2.find(p);
        if(pinv2.end() == invIt)
        {
          merge[p.first] = p.second;
          maxHierarchy[p.first].persistence = pers;
          maxHierarchy[p.first].parent = p.second;
          maxHierarchy[p.first].saddle = saddleIdx;

          ptmp[std::pair<T,int>(pers,saddleIdx)] = p;
          pinv2[p] = std::pair<T,int>(pers,saddleIdx);
        }
      }
    }
    else if (this->persistenceType.compare("count") == 0)
    {
      int newPersistence = 0;
      int oldPersistence = 0;
      for(int idx = 0; idx < Size(); idx++)
      {
        int extIdx = followChain(flow[idx], merge);
        if (extIdx == p.first)
          newPersistence++;
        if (extIdx == pold.first)
          oldPersistence++;
      }

      //check if there is new merge pair with increased persistence (or same
      // persistence and a larger index maximum)
      T diff = newPersistence - oldPersistence;
      if( diff > 0 || (diff == 0 && p.first > pold.first ))
      {
        //if the persistence increased insert into the persistence list and
        //merge possible other extrema with smaller persistence values first
        double npers = newPersistence;
        persistence[std::pair<T,int>(npers,saddleIdx)] = p;
      }
      //otherwise merge the pair
      else
      {
        //check if the pair has not been previously merged
        map_pi_pfi_it invIt = pinv2.find(p);
        if(pinv2.end() == invIt)
        {
          merge[p.first] = p.second;
          maxHierarchy[p.first].persistence = pers;
          maxHierarchy[p.first].parent = p.second;
          maxHierarchy[p.first].saddle = saddleIdx;

          ptmp[std::pair<T,int>(pers,saddleIdx)] = p;
          pinv2[p] = std::pair<T,int>(pers,saddleIdx);
        }
      }
    }
    else if (this->persistenceType.compare("area") == 0)
    {
      //FIXME: implement this & test
    }
  }

  DebugPrint("pers saddleIdx : merged -> parent\n");
  if (globalVerbosity)
  {
    for(map_pi_pfi_it it = pinv2.begin(); it != pinv2.end(); it++)
    {
      std::stringstream ss;
      ss << (*it).second.first << " " << (*it).second.second << ":"
         << (*it).first.first << " -> " << (*it).first.second << std::endl;
      DebugPrint(ss.str());
    }
  }
}

template<typename T>
MorseComplex<T>::MorseComplex(std::vector<T> &Xin,
              std::vector<T> &yin,
              std::string gradientMethod,
              std::string persistenceType,
              std::vector<T> &win,
              std::map< int, std::set<int> > &neighborhoods,
              bool verbosity)
{
  this->persistenceType = persistenceType;
  globalVerbosity = verbosity;
  time_t myTime;
  DebugTimerStart(myTime, "\rInitializing...");

  int M = Xin.size() / yin.size();
  int N = yin.size();

  X = std::vector< std::vector<T> >(M,std::vector<T>(N,0));
  y = yin;
  w = win;

  T sumW = 0;
  for(int n = 0; n < N; n++)
  {
    for(int m = 0; m < M; m++)
      X[m][n] = Xin[n*M+m];

    sumW += w[n];
  }

  if (sumW > 0)
    for(int n = 0; n < N; n++)
    {
      w[n] /= (T) sumW;
    }

  neighbors = neighborhoods;

  DebugTimerStop(myTime);

  DebugTimerStart(myTime, "\rComputing distances...");
  computeDistances();
  DebugTimerStop(myTime);
  DebugTimerStart(myTime, "\rEstimating integral lines...");
  EstimateIntegralLines(gradientMethod);
  DebugTimerStop(myTime);
  DebugTimerStart(myTime, "\rComputing persistence for maxima...\n");
  ComputeMaximaPersistence();
  DebugTimerStop(myTime);
  DebugTimerStart(myTime, "\rCleaning up...");
  DebugTimerStop(myTime);
  DebugPrint("\rMy work is complete. The Maker would be pleased.");

  // Always reset the global verbosity to false
  globalVerbosity = false;
}

template<typename T>
void MorseComplex<T>::computeDistances()
{
  distances.clear();

  int i, j;
  std::set<int> Ni;
  int dims = Dimension();

  for ( std::map<int, std::set<int> >:: iterator it = neighbors.begin();
        it != neighbors.end();
        it++ )
  {
      i = it->first;
      Ni = it->second;

      for (std::set<int>::iterator it = Ni.begin(); it != Ni.end(); it++)
      {
        j = *it;
        T dist = 0;
        for(int d = 0; d < dims; d++)
            dist += ((X[d][i]-X[d][j])*(X[d][i]-X[d][j]));

        distances[sortedPair(i,j)] = sqrt(dist);
      }
  }
}

//Look-up Operations

template<typename T>
int MorseComplex<T>::Dimension()
{
  return (int) X.size();
}

template<typename T>
int MorseComplex<T>::Size()
{
  return y.size();
}

template<typename T>
void MorseComplex<T>::GetX(int i, T *xi)
{
  for(int d = 0; d < Dimension(); d++)
    xi[d] = X[d][i];
}

template<typename T>
T MorseComplex<T>::GetX(int i, int j)
{
  return X[i][j];
}

template<typename T>
T MorseComplex<T>::GetY(int i)
{
  return y[i];
}

//Computed Quantities

template<typename T>
T MorseComplex<T>::MinY()
{
  T minY = y[0];
  for(int i = 1; i < Size(); i++)
    minY = minY > y[i] ? y[i] : minY;
  return minY;
}

template<typename T>
T MorseComplex<T>::MaxY()
{
  T maxY = y[0];
  for(int i = 1; i < Size(); i++)
    maxY = maxY < y[i] ? y[i] : maxY;
  return maxY;
}

template<typename T>
T MorseComplex<T>::RangeY()
{
  return MaxY()-MinY();
}

template<typename T>
T MorseComplex<T>::MinX(int dim)
{
  T minX = X[dim][0];
  for(int i = 1; i < Size(); i++)
    minX = minX > X[dim][i] ? X[dim][i] : minX;
  return minX;
}

template<typename T>
T MorseComplex<T>::MaxX(int dim)
{
  T maxX = X[dim][0];
  for(int i = 1; i < Size(); i++)
    maxX = maxX < X[dim][i] ? X[dim][i] : maxX;
  return maxX;
}

template<typename T>
T MorseComplex<T>::RangeX(int dim)
{
  return MaxX(dim)-MinX(dim);
}

template<typename T>
int MorseComplex<T>::MaxLabel(int i, T pers)
{
  int maxIdx = flow[i];
  while(maxHierarchy[maxIdx].persistence < pers)
    maxIdx = maxHierarchy[maxIdx].parent;
  return maxIdx;
}

template<typename T>
std::string MorseComplex<T>::to_json()
{
  persistence_map_it it;
  std::stringstream stream;

  stream << "{\"Hierarchy\":[";
  for(it = maxHierarchy.begin(); it != maxHierarchy.end(); it++)
  {
    if (it != maxHierarchy.begin()) {
        stream << ",";
    }
    stream << "{\"Persistence\":" << it->second.persistence << ",\"Dying\":"
           << it->first << ",\"Surviving\":" << it->second.parent
           << ",\"Saddle\":" << it->second.saddle << "}";
  }
  stream << "],\"Partitions\":[";
  for(std::vector<int>::iterator i = flow.begin(); i != flow.end(); i++)
  {
      if (i != flow.begin()) {
        stream << ",";
      }
      stream << *i;
  }
  stream << "]}";
  return stream.str();
}

template<typename T>
std::vector<T> MorseComplex<T>::SortedPersistences()
{
  std::set<T> setP;
  std::vector<T> sortedP;
  for(persistence_map_it it = maxHierarchy.begin(); it != maxHierarchy.end();it++)
    setP.insert(it->second.persistence);
  std::copy(setP.begin(), setP.end(), std::back_inserter(sortedP));
  std::sort (sortedP.begin(), sortedP.end());
  return sortedP;
}

template<typename T>
std::map< int, std::vector<int> > MorseComplex<T>::GetPartitions(T persistence)
{
  T minP = SortedPersistences()[0];

  std::map< int, std::vector<int> > partitions;
  for(int i = 0; i < Size(); i++)
  {
    int maxIdx = MaxLabel(i,minP);

    while(maxHierarchy[maxIdx].persistence < persistence
          && maxIdx != maxHierarchy[maxIdx].parent)
    {
      maxIdx = maxHierarchy[maxIdx].parent;
    }
    if( partitions.find(maxIdx) == partitions.end())
    {
      partitions[maxIdx] = std::vector<int>();
      partitions[maxIdx].push_back(maxIdx);
    }

    if(i != maxIdx)
      partitions[maxIdx].push_back(i);
  }

  return partitions;
}

template class MorseComplex<double>;
template class MorseComplex<float>;
