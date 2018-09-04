#include "UnionFind.h"

UnionFind::UnionFind() { }

UnionFind::~UnionFind()
{
  for(std::map<int, Singleton *>::iterator iter = sets.begin();
    iter != sets.end();
    iter++) {
    delete iter->second;
  }
}

void UnionFind::MakeSet(int id)
{
  if(sets.find(id) != sets.end())
  {
    std::cerr << "ERROR: Singleton " << id << " already exists" << std::endl;
    return;
  }
  sets[id] = new Singleton(id);
}

int UnionFind::Find(int id)
{
  if(sets.find(id) == sets.end())
    MakeSet(id);

  if(sets[id]->parent == id)
    return id;
  else
    return sets[id]->parent = Find(sets[id]->parent);
}

void UnionFind::Union(int x, int y)
{
  int xRoot = Find(x);
  int yRoot = Find(y);
  if( xRoot == yRoot)
    return;

  if( sets[xRoot]->rank < sets[yRoot]->rank
   || (sets[xRoot]->rank < sets[yRoot]->rank && xRoot < yRoot) )
  {
    sets[xRoot]->parent = yRoot;
    sets[yRoot]->rank = sets[yRoot]->rank + 1;
  }
  else
  {
    sets[yRoot]->parent = xRoot;
    sets[xRoot]->rank = sets[xRoot]->rank + 1;
  }
}

int UnionFind::CountComponents()
{
  int count = 0;
  std::set<int> roots;
  for(std::map<int, Singleton *>::iterator iter = sets.begin();
      iter != sets.end();
      iter++)
  {
    int root = Find(iter->first);
    if( roots.find(root) == roots.end())
    {
      roots.insert(root);
      count++;
    }
  }
  return count;
}

void UnionFind::GetComponentRepresentatives(std::vector<int> &reps)
{
  int count = 0;
  std::set<int> roots;
  for(std::map<int, Singleton *>::iterator iter = sets.begin();
      iter != sets.end();
      iter++)
  {
    int root = Find(iter->first);
    if( roots.find(root) == roots.end())
    {
      roots.insert(root);
      reps.push_back(root);
      count++;
    }
  }
}

void UnionFind::GetComponentItems(int rep, std::vector<int> &items)
{
  int count = 0;
  for(std::map<int, Singleton *>::iterator iter = sets.begin();
      iter != sets.end();
      iter++)
  {
    int root = Find(iter->first);
    if(rep == root)
    {
      items.push_back(iter->first);
      count++;
    }
  }
}
