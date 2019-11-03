#pragma once

#include <chrono>
#include <string>

namespace timing {

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

class TicTocBank {
 public:
  TicTocBank();

  void Reset();
  void Update(const std::string &name, int64_t duration_ns);
};

class TicToc {
 public:
  explicit TicToc(const std::string &name);
  
  void Tic();
  void Toc();
 
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::string name_;
};

} // namespace timing
