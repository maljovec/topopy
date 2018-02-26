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

#include "AMSC.h"
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
int AMSC<T>::ascending(int index)
{
  return neighborFlow[index].up;
}

template<typename T>
void AMSC<T>::SteepestEdge()
{
  for( int i = 0; i < Size(); i++)
    neighborFlow.push_back(FlowPair(-1,-1));

  //Store the gradient magnitude of each point's largest ascent/descent
  // so we can verify if the next neighbor represents a larger jump.
  std::vector< std::vector<T> > G = std::vector< std::vector<T> >(2, std::vector<T>(Size(), 0));

  //compute steepest asc/descending neighbors
  for(int i=0; i < (int)Size(); i++)
  {
    std::set<int> Ni = neighbors[i];
    int j; // An index adjacent to i
    
    for (std::set<int>::iterator it = Ni.begin(); it != Ni.end(); it++)
    {
      j = *it;

      double g = (y[j] - y[i]) / distances[sortedPair(i,j)];
      //Compare to i's neighborhood
      if(G[0][i] < g)
      {
        //j is a steeper ascent than current
        G[0][i] = g;
        neighborFlow[i].up = j;
      }
      else if(G[0][i] == g
           && neighborFlow[i].up != -1
           && neighborFlow[i].up < j)
      {
        //j is as steep an ascent as current,
        // and j is a larger index than current
        G[0][i] = g;
        neighborFlow[i].up = j;
      }
      else if(G[0][i] == g
          && neighborFlow[i].up == -1
          && i < j)
      {
        //j is as steep an ascent as current,
        // and current is not set and j is larger than i
        G[0][i] = g;
        neighborFlow[i].up = j;
      }
      else if(G[1][i] > g)
      {
        //j is a steeper descent than current
        G[1][i] = g;
        neighborFlow[i].down = j;
      }
      else if(G[1][i] == g
           && neighborFlow[i].down != -1
           && neighborFlow[i].down > j)
      {
        //j is as steep a descent as current,
        // and j is a smaller index than current
        G[1][i] = g;
        neighborFlow[i].down = j;
      }
      else if(G[1][i] == g
           && neighborFlow[i].down == -1
           && i > j)
      {
        //j is as steep a descent as current,
        // and the current is not set and j is smaller than i
        G[1][i] = g;
        neighborFlow[i].down = j;
      }

      //Look to j's neighborhood
      if(G[0][j] < -g)
      {
        //i is a steeper ascent than current
        G[0][j] = -g;
        neighborFlow[j].up = i;
      }
      else if(G[0][j] == -g
           && neighborFlow[j].up != -1
           && neighborFlow[j].up < i)
      {
        //i is as steep an ascent as current,
        // and i is a larger index than current
        G[0][j] = -g;
        neighborFlow[j].up = i;
      }
      else if(G[0][j] == -g
           && neighborFlow[j].up == -1
           && j < i)
      {
        //i is as steep an ascent as current,
        // and current is not set and i is larger than j
        G[0][j] = -g;
        neighborFlow[j].up = i;
      }
      else if(G[1][j] > -g)
      {
        //i is a steeper descent than current
        G[1][j] = -g;
        neighborFlow[j].down = i;
      }
      else if(G[1][j] == -g
           && neighborFlow[j].down != -1
           && neighborFlow[j].down > i)
      {
        //i is as steep a descent as current,
        // and i is a smaller index than current
        G[1][j] = -g;
        neighborFlow[j].down = i;
      }
      else if(G[1][j] == -g
           && neighborFlow[j].down == -1
           && j > i)
      {
        //i is as steep a descent as current,
        // and current is not set and i is smaller than j
        G[1][j] = -g;
        neighborFlow[j].down = i;
      }
    }
  }

  //compute for each point its minimum and maximum based on
  //steepest ascent/descent
  for(int i = 0; i < Size(); i++)
    flow.push_back(FlowPair(-1,-1));

  std::list<int> path;
  for(int i=0; i < Size(); i++)
  {
    //If we have not identified this point's maximum, then we will do so now
    if( flow[i].up == -1)
    {
      //Recursively trace the upward flow from this point along path, until
      // we reach a point that has no upward flow
      path.clear();
      int prev = i;
      while(prev != -1 && flow[prev].up == -1)
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
        ext = flow[prev].up;

      for(std::list<int>::iterator it = path.begin(); it!=path.end(); ++it)
        flow[*it].up = ext;
    }
  }

  for(int i=0; i < Size(); i++)
  {
    if( flow[i].down == -1)
    {
      path.clear();
      int prev = i;
      while(prev != -1 && flow[prev].down == -1)
      {
        path.push_back(prev);
        prev = descending(prev);
      }
      int ext = -1;
      if(prev == -1)
      {
        ext = path.back();
        if(this->persistenceType.compare("difference") == 0)
        {
          minHierarchy[ext] = Merge<T>(RangeY(),ext,ext);
        }
        else if(this->persistenceType.compare("count") == 0)
        {
          minHierarchy[ext] = Merge<T>(Size(),ext,ext);
        }
        else if(this->persistenceType.compare("probability") == 0)
        {
          minHierarchy[ext] = Merge<T>(1,ext,ext);
        }
      }
      else
        ext = flow[prev].down;

      for(std::list<int>::iterator it = path.begin(); it!=path.end(); ++it)
        flow[*it].down = ext;
    }
  }
}

//TODO: Repeat the process for the negative gradient, and then figure out how
// to do the probabilistic trace in a Markov chain.
template<typename T>
void AMSC<T>::MaxFlow()
{
  for( int i = 0; i < Size(); i++)
    neighborFlow.push_back(FlowPair(-1,-1));

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
        neighborFlow[i].up = j;
        //repeat for neighborFlow[i].down
        //neighborFlow[i].down = j;
      }
      actualNeighborCount++;
    }
    delete [] probability;
  }

  //Must now compute flow from neighborflow

  delete [] avgGradient;
  delete [] neighborGradient;
}

template<typename T>
void AMSC<T>::EstimateIntegralLines(std::string method)
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
void AMSC<T>::ComputeMaximaPersistence()
{
  //initial persistences
  //store as pairs of extrema such that p.first merges to p.second (e.g.
  //p.second is the max with the larger function value
  map_pi_pfi pinv;
  for(int i = 0; i < Size(); i++)
  {
    int e1 = flow[i].up;
    int saddleIdx;

    std::set<int> Ni = neighbors[i];
    int j; // An index adjacent to i
    
    for (std::set<int>::iterator it = Ni.begin(); it != Ni.end(); it++)
    {
      j = *it;

      int e2 = flow[j].up;
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
            if (flow[idx].up == p.first)
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
            if (flow[idx].up == p.first)
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
        int extIdx = followChain(flow[idx].up, merge);
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
        int extIdx = followChain(flow[idx].up, merge);
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
void AMSC<T>::ComputeMinimaPersistence()
{
  //initial persistences
  //store as pairs of extrema such that p.first merges to p.second (e.g.
  //p.second is the min with the smaller function value
  map_pi_pfi pinv;
  for(int i = 0; i < Size(); i++)
  {
    int saddleIdx;
    int e1 = flow[i].down;
    std::set<int> Ni = neighbors[i];
    int j; // An index adjacent to i
    
    for (std::set<int>::iterator it = Ni.begin(); it != Ni.end(); it++)
    {
      j = *it;

      int e2 = flow[j].down;
      if(e1 != e2)
      {
        T pers = 0;
        //p should have the merging index in the first position and the parent
        // in the second
        int_pair p(e1,e2);
        if( y[e1] < y[e2] || (y[e1] == y[e2] && e1 < e2) )
        {
          p.first = e2;
          p.second = e1;
        }

        saddleIdx = y[i] > y[j] ? i : j;
        if (this->persistenceType.compare("difference") == 0)
        {
          pers = y[saddleIdx] - y[p.first];
        }
        else if (this->persistenceType.compare("probability") == 0)
        {
          T probabilityIntegral = 0;
          int count = 0;

          for(int idx = 0; idx < Size(); idx++)
          {
            if (flow[idx].down == p.first)
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
          int count = 0;
          for(int idx = 0; idx < Size(); idx++)
            if (flow[idx].down == p.first)
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
            minHierarchy[p.first].persistence = pers;
            minHierarchy[p.first].parent = p.second;
            minHierarchy[p.first].saddle = saddleIdx;
          }
        }
        else
        {
          pinv[p] = std::pair<T,int>(pers,saddleIdx);
          minHierarchy[p.first].persistence = pers;
          minHierarchy[p.first].parent = p.second;
          minHierarchy[p.first].saddle = saddleIdx;
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

  //First, set each minimum to merge into itself
  std::map<int,int> merge;
  for(persistence_map_it it = minHierarchy.begin();
      it != minHierarchy.end();
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
    p.first = followChain(p.first, merge);
    p.second = followChain(p.second, merge);

    if( y[p.first] < y[p.second] )
      std::swap(p.second, p.first);

    //remove current merge pair from list
    persistence.erase(it);

    //are the extrema already merged?
    if(p.first == p.second)
      continue;

    //check if there is new merge pair with increased persistence (or same
    // persistence and a smaller index minimum)
    if (this->persistenceType.compare("difference") == 0)
    {
      T diff = y[pold.first] - y[p.first];
      if( diff > 0 || (diff == 0 && p.first < pold.first ))
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
          minHierarchy[p.first].persistence = pers;
          minHierarchy[p.first].parent = p.second;
          minHierarchy[p.first].saddle = saddleIdx;
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
        int extIdx = followChain(flow[idx].down, merge);
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
        newPersistence /= (T) newCount;
      if (oldCount > 0)
        oldPersistence /= (T) oldCount;


      T diff = newPersistence - oldPersistence;
      if( diff > 0 || (diff == 0 && p.first < pold.first ))
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
          minHierarchy[p.first].persistence = pers;
          minHierarchy[p.first].parent = p.second;
          minHierarchy[p.first].saddle = saddleIdx;
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
        int extIdx = followChain(flow[idx].down, merge);
        if (extIdx == p.first)
          newPersistence++;
        if (extIdx == pold.first)
          oldPersistence++;
      }

      T diff = newPersistence - oldPersistence;
      if( diff > 0 || (diff == 0 && p.first < pold.first ))
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
          minHierarchy[p.first].persistence = pers;
          minHierarchy[p.first].parent = p.second;
          minHierarchy[p.first].saddle = saddleIdx;
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
AMSC<T>::AMSC(std::vector<T> &Xin,
              std::vector<T> &yin,
              std::vector<std::string> &_names,
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

  for(unsigned int i = 0; i < _names.size(); i++)
    names.push_back(_names[i]);

  int M = Xin.size() / yin.size();
  int N = yin.size();

  X = std::vector< std::vector<T> >(M,std::vector<T>(N,0));
  y = yin;
  w = win;

  globalMinIdx = 0;
  globalMaxIdx = 0;
  T sumW = 0;
  for(int n = 0; n < N; n++)
  {
    for(int m = 0; m < M; m++)
      X[m][n] = Xin[n*M+m];

    sumW += w[n];

    if(y[n] > y[globalMaxIdx])
      globalMaxIdx = n;
    if(y[n] < y[globalMinIdx])
      globalMinIdx = n;
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
  DebugTimerStart(myTime, "\rComputing persistence for minima...\n");
  ComputeMinimaPersistence();
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
void AMSC<T>::computeDistances()
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
int AMSC<T>::Dimension()
{
  return (int) X.size();
}

template<typename T>
int AMSC<T>::Size()
{
  return y.size();
}

template<typename T>
void AMSC<T>::GetX(int i, T *xi)
{
  for(int d = 0; d < Dimension(); d++)
    xi[d] = X[d][i];
}

template<typename T>
T AMSC<T>::GetX(int i, int j)
{
  return X[i][j];
}

template<typename T>
T AMSC<T>::GetY(int i)
{
  return y[i];
}

template<typename T>
std::string AMSC<T>::Name(int dim)
{
  return names[dim];
}

//Computed Quantities

template<typename T>
T AMSC<T>::MinY()
{
  T minY = y[0];
  for(int i = 1; i < Size(); i++)
    minY = minY > y[i] ? y[i] : minY;
  return minY;
}

template<typename T>
T AMSC<T>::MaxY()
{
  T maxY = y[0];
  for(int i = 1; i < Size(); i++)
    maxY = maxY < y[i] ? y[i] : maxY;
  return maxY;
}

template<typename T>
T AMSC<T>::RangeY()
{
  return MaxY()-MinY();
}

template<typename T>
T AMSC<T>::MinX(int dim)
{
  T minX = X[dim][0];
  for(int i = 1; i < Size(); i++)
    minX = minX > X[dim][i] ? X[dim][i] : minX;
  return minX;
}

template<typename T>
T AMSC<T>::MaxX(int dim)
{
  T maxX = X[dim][0];
  for(int i = 1; i < Size(); i++)
    maxX = maxX < X[dim][i] ? X[dim][i] : maxX;
  return maxX;
}

template<typename T>
T AMSC<T>::RangeX(int dim)
{
  return MaxX(dim)-MinX(dim);
}

template<typename T>
int AMSC<T>::MinLabel(int i, T pers)
{
  int minIdx = flow[i].down;
  while(minHierarchy[minIdx].persistence < pers)
    minIdx = minHierarchy[minIdx].parent;
  return minIdx;
}

template<typename T>
int AMSC<T>::MaxLabel(int i, T pers)
{
  int maxIdx = flow[i].up;
  while(maxHierarchy[maxIdx].persistence < pers)
    maxIdx = maxHierarchy[maxIdx].parent;
  return maxIdx;
}

template<typename T>
std::string AMSC<T>::PrintHierarchy()
{
  persistence_map_it it;
  std::stringstream stream;
  char sep = ',';

  for(it  = minHierarchy.begin(); it != minHierarchy.end(); it++)
    stream << "Minima" << sep << it->second.persistence << sep
           << it->first << sep << it->second.parent << ' ';

  for(it = maxHierarchy.begin(); it != maxHierarchy.end(); it++)
    stream << "Maxima" << sep << it->second.persistence << sep
           << it->first << sep << it->second.parent << ' ';

  return stream.str();
}

template<typename T>
std::string AMSC<T>::XMLFormattedHierarchy()
{
  persistence_map_it it;
  std::stringstream stream;
  for(it  = minHierarchy.begin(); it != minHierarchy.end(); it++)
  {

    stream << "<Minimum";
    if(it->first == it->second.parent)
      stream << " global=\"True\"";
    stream << ">" << std::endl << "\t<id>" << it->first << "</id>" << std::endl;
    if(it->first != it->second.parent)
      stream << "\t<target>" << it->second.parent << "</target>" << std::endl;
    stream << "\t<persistence>" << it->second.persistence << "</persistence>"
           << std::endl
           << "</Minimum>" << std::endl;
  }
  for(it = maxHierarchy.begin(); it != maxHierarchy.end(); it++)
  {
    stream << "<Maximum";
    if(it->first == it->second.parent)
      stream << " global=\"True\"";
    stream << ">" << std::endl << "\t<id>" << it->first << "</id>" << std::endl;
    if(it->first != it->second.parent)
      stream << "\t<target>" << it->second.parent << "</target>" << std::endl;
    stream << "\t<persistence>" << it->second.persistence << "</persistence>"
           << std::endl
           << "</Maximum>" << std::endl;
  }
  return stream.str();

}

template<typename T>
std::vector<T> AMSC<T>::SortedPersistences()
{
  std::set<T> setP;
  std::vector<T> sortedP;
  for(persistence_map_it it = minHierarchy.begin(); it != minHierarchy.end();it++)
    setP.insert(it->second.persistence);
  for(persistence_map_it it = maxHierarchy.begin(); it != maxHierarchy.end();it++)
    setP.insert(it->second.persistence);
  std::copy(setP.begin(), setP.end(), std::back_inserter(sortedP));
  std::sort (sortedP.begin(), sortedP.end());
  return sortedP;
}

template<typename T>
std::map< std::string, std::vector<int> > AMSC<T>::GetPartitions(T persistence)
{
  T minP = SortedPersistences()[0];

  std::map< std::string, std::vector<int> > partitions;
  for(int i = 0; i < Size(); i++)
  {
    std::stringstream stream;
    int minIdx = MinLabel(i,minP);
    int maxIdx = MaxLabel(i,minP);

    while(minHierarchy[minIdx].persistence < persistence
          && minIdx != minHierarchy[minIdx].parent)
    {
      minIdx = minHierarchy[minIdx].parent;
    }

    while(maxHierarchy[maxIdx].persistence < persistence
          && maxIdx != maxHierarchy[maxIdx].parent)
    {
      maxIdx = maxHierarchy[maxIdx].parent;
    }

    stream << minIdx << ',' << maxIdx;
    std::string label = stream.str();
    if( partitions.find(label) == partitions.end())
    {
      partitions[label] = std::vector<int>();
      partitions[label].push_back(minIdx);
      partitions[label].push_back(maxIdx);
    }

    if(i != minIdx && i != maxIdx)
      partitions[label].push_back(i);
  }

  return partitions;
}

template<typename T>
std::map< int, std::vector<int> > AMSC<T>::GetStableManifolds(T persistence)
{
  T minP = SortedPersistences()[0];

  std::map< int, std::vector<int> > partitions;
  for(int i = 0; i < Size(); i++)
  {
    int minIdx = MinLabel(i,minP);
    int maxIdx = MaxLabel(i,minP);

    while(maxHierarchy[maxIdx].persistence < persistence
          && maxIdx != maxHierarchy[maxIdx].parent)
    {
      maxIdx = maxHierarchy[maxIdx].parent;
    }
    if( partitions.find(maxIdx) == partitions.end())
    {
      partitions[maxIdx] = std::vector<int>();
      partitions[maxIdx].push_back(minIdx);
      partitions[maxIdx].push_back(maxIdx);
    }

    if(i != minIdx && i != maxIdx)
      partitions[maxIdx].push_back(i);
  }

  return partitions;
}

template<typename T>
std::map< int, std::vector<int> > AMSC<T>::GetUnstableManifolds(T persistence)
{
  T minP = SortedPersistences()[0];

  std::map< int, std::vector<int> > partitions;
  for(int i = 0; i < Size(); i++)
  {
    std::stringstream stream;
    int minIdx = MinLabel(i,minP);
    int maxIdx = MaxLabel(i,minP);

    while(minHierarchy[minIdx].persistence < persistence
          && minIdx != minHierarchy[minIdx].parent)
    {
      minIdx = minHierarchy[minIdx].parent;
    }

    if( partitions.find(minIdx) == partitions.end())
    {
      partitions[minIdx] = std::vector<int>();
      partitions[minIdx].push_back(minIdx);
      partitions[minIdx].push_back(maxIdx);
    }

    if(i != minIdx && i != maxIdx)
      partitions[minIdx].push_back(i);
  }

  return partitions;
}

template<typename T>
std::set<int> AMSC<T>::Neighbors(int index)
{
  return neighbors[index];
}

template class AMSC<double>;
template class AMSC<float>;
