#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

#include <lcmtypes/timing/tictoc_t.hpp>

#define kSecPerNanoSec 1e-9

/* @see utils/tests/timing_test.cc for detailed usage */
#define TIC_TOC_SCOPE(name) timing::ScopedTicToc scoped_tictoc_##name(#name)
#define TIC_TOC_FUNCTION() timing::ScopedTicToc function_scoped_tictoc(__FUNCTION__)

#define TIC(name) timing::TicToc local_tictoc_##name(#name)
#define TOC(name) local_tictoc_##name.Toc()

namespace timing {

typedef std::chrono::duration<int64_t, std::milli> milli_sec_t;
typedef std::chrono::duration<int64_t, std::micro> micro_sec_t;
typedef std::chrono::duration<int64_t, std::nano> nano_sec_t;

/***********************
 * --- TicTocStats --- *
 ***********************/

/**
 * @brief timing statistics for one profiled target
 */
class TicTocStats : public tictoc_stats_t {
 public:
  TicTocStats();
  TicTocStats(tictoc_stats_t &stats);

  void Update(int64_t duration_ns);
  void Reset();
  
  double TotalTime();
  double AverageTime();
  double MaxTime();
  double MinTime();
  int32_t Count();
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

  /**
   * @brief get tic toc summary as lcm data struct
   *
   * @return tic toc lcm data struct
   */
  tictoc_t GetLCM();
 
 private:
  std::unordered_map<std::string, TicTocStats> channel_map_;
  std::mutex lock_;
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
  std::chrono::time_point<std::chrono::steady_clock> start_;
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
  std::chrono::time_point<std::chrono::steady_clock> start_;
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
