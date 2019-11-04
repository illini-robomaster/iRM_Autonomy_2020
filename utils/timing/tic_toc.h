#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

#define kSecPerNanoSec 1e-9

/* @see utils/tests/timing_test.cc for detailed usage */
#define TIC_TOC_SCOPE(name) ScopedTicToc scoped_tictoc_##name(#name)
#define TIC_TOC_FUNCTION() ScopedTicToc function_scoped_tictoc(__FUNCTION__)

namespace timing {

/***********************
 * --- TicTocStats --- *
 ***********************/

/**
 * @brief timing statistics for one profiled target
 */
class TicTocStats {
 public:
  TicTocStats();

  void Update(int64_t duration_ns);
  void Reset();
  
  double TotalTime();
  double AverageTime();
  double MaxTime();
  double MinTime();
  int32_t Count();
 
 private:
  int32_t num_stats_;
  int64_t min_time_ns_;
  int64_t max_time_ns_;
  int64_t total_time_ns_;
};

/**********************
 * --- TicTocBank --- *
 **********************/

/**
 * @brief A bank of tic toc statistics for all profiled instances
 */
class TicTocBank {
 public:
  TicTocBank();

  /**
   * @brief reset entire bank
   */
  void Reset();

  /**
   * @brief update profile statistics for a given channel
   *
   * @param name        channel name
   * @param duration_ns time duration in nano seconds
   */
  void Update(const std::string &name, int64_t duration_ns);
  std::unordered_map<std::string, TicTocStats> GetSummary();
 
 private:
  std::unordered_map<std::string, TicTocStats> channel_map_;
};

/******************
 * --- TicToc --- *
 ******************/

/**
 * @brief simple tic toc profiler
 *
 * @note example usage:
 *    timing::TicToc tictoc_x(<channel_name>);
 *
 *    tictoc_x.Tic(); // this line is optional as it is called in the constructor
 *    some_time_consuming_function();
 *    tictoc_x.Toc();
 */
class TicToc {
 public:
   /**
    * @brief constructor
    *
    * @param name channel name
    *
    * @remark calls Tic() to start the timer when constructing
    */
  explicit TicToc(const std::string &name);
  
  /**
   * @brief start the timer
   */
  void Tic();

  /**
   * @brief end the timer and upload duration to a global manager
   */
  void Toc();
 
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::string name_;
};

/**
 * @brief Simple profiler for timing a c++ scope
 */
class ScopedTicToc {
 public:
   /**
    * @brief constructor
    *
    * @param name channel name
    *
    * @remark start timing at construction
    */
  explicit ScopedTicToc(const std::string &name);

  /**
   * @brief destructor when object goes out of scope
   *
   * @remark end the timer and upload duration to a global manager
   */
  ~ScopedTicToc();

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::string name_;
};

/**************************
 * --- Global Context --- *
 **************************/

/**
 * @brief upload profiling statistics to a global channel
 *
 * @param name        channel name
 * @param duration_ns timing duration in nano second
 */
void TicTocGlobalUpdate(const std::string &name, int64_t duration_ns);

/**
 * @brief clear all global channel statistics
 */
void TicTocGlobalReset();

/**
 * @brief get a string representation of all global channel timing statistics
 *
 * @return global channel timing statistics in the format of
 *    | Channel Name | Count | Total Time | Min Time | Average Time | Max Time |
 */
std::string TicTocGlobalSummary();

} // namespace timing
