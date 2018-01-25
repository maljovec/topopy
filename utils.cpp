#include "utils.h"
#include <iostream>
#include <sstream>

void ToggleVerbosity(bool on)
{
  globalVerbosity = on;
}

void DebugPrint(std::string text)
{
  if(!globalVerbosity)
    return;
  std::cerr << text << std::flush;
}

void DebugTimerStart(time_t &t0, std::string text)
{
  if(!globalVerbosity)
    return;
  t0 = clock();
  std::cerr << text << std::flush;
}

void DebugTimerStop(time_t &t0, std::string text)
{
  if(!globalVerbosity)
    return;
  time_t endTime = clock();
  std::stringstream ss;
  // Possibly use text here, for now it works since each DebugTimerStart will
  // be followed by a DebugTimerEnd, and thus the combined output can be printed
  // on one line
  std::cerr << "Done!" << " (" << ((float)endTime-t0)/CLOCKS_PER_SEC << "s)"
            << std::endl;
}

std::pair<int, int> sortedPair(int a, int b)
{
  std::pair<int,int> myPair;
  if (a < b)
  {
    myPair = std::pair<int,int>(a, b);
  }
  else
  {
    myPair = std::pair<int,int>(b, a);
  }
  return myPair;
}