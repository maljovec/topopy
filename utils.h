/******************************************************************************
 * Software License Agreement (BSD License)                                   *
 *                                                                            *
 * Copyright 2016 University of Utah                                          *
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