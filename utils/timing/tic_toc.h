#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

#define kSecPerNanoSec 1e-9

namespace timing {

/***********************
 * --- TicTocStats --- *
 ***********************/

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

class TicTocBank {
 public:
  TicTocBank();

  void Reset();
  void Update(const std::string &name, int64_t duration_ns);
  std::unordered_map<std::string, TicTocStats> GetSummary();
 
 private:
  std::unordered_map<std::string, TicTocStats> channel_map_;
};

/******************
 * --- TicToc --- *
 ******************/

class TicToc {
 public:
  explicit TicToc(const std::string &name);
  
  void Tic();
  void Toc();
 
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::string name_;
};

class ScopedTicToc {
 public:
  explicit ScopedTicToc(const std::string &name);
  ~ScopedTicToc();

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::string name_;
};

#define TIC_TOC_SCOPE(name) ScopedTicToc scoped_tictoc_##name(#name)
#define TIC_TOC_FUNCTION() ScopedTicToc function_scoped_tictoc(__FUNCTION__)

/**************************
 * --- Global Context --- *
 **************************/

void TicTocGlobalUpdate(const std::string &name, int64_t duration_ns);
void TicTocGlobalReset();
std::string TicTocGlobalSummary();

} // namespace timing
