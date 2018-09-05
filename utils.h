#ifndef UTILS_H
#define UTILS_H

#include <ctime>
#include <string>
#include <utility>

static bool globalVerbosity = false;
/**
 * Helper method that optionally prints a message depending on the verbosity
 * of this object
 * @param text string of the message to display
 */
void DebugPrint(std::string text);

/**
 * Helper method that optionally starts a timer and prints a message depending
 * on the verbosity of this object
 * @param t0 reference time_t struct that will be written to
 * @param text string of message to display onscreen
 */
void DebugTimerStart(time_t &t0, std::string text);

/**
 * Helper method that optionally stops a timer and prints a message depending
 * on the verbosity of this object
 * @param t0 reference time_t struct that will be used as the start time
 */
void DebugTimerStop(time_t &t0, std::string text="");

/**
 * Method for turning on or off the verbosity
 * @param on boolean representing the new state of the globalVerbosity variable
 */
void ToggleVerbosity(bool on);

/**
 * Helper method that creates a pair of integers in sorted order from two
 * arbitrarily sorted incoming values
 * @param a integer to order
 * @param b integer to order
 */
std::pair<int, int> sortedPair(int a, int b);

#endif //UTILS_H