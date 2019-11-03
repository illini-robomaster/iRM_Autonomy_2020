#include "utils/timing/tic_toc.h"

using namespace std;

namespace timing {

/***********************
 * --- TicTocStats --- *
 ***********************/

TicTocStats::TicTocStats() { Reset(); }

void TicTocStats::Reset() {
  num_stats_ = 0;
  min_time_ns_ = std::numeric_limits<int64_t>::max();
  max_time_ns_ = std::numeric_limits<int64_t>::min();
  total_time_ns_ = 0;
}

void TicTocStats::Update(int64_t duration_ns) {
  ++num_stats_;
  min_time_ns_ = std::min(min_time_ns_, duration_ns);
  max_time_ns_ = std::max(max_time_ns_, duration_ns);
  total_time_ns_ += duration_ns;
}

double TicTocStats::TotalTime() {
  return static_cast<double>(total_time_ns_);
}

double TicTocStats::AverageTime() {
  return TotalTime() / num_stats_;
}

double TicTocStats::MaxTime() {
  return static_cast<double>(max_time_ns_);
}

double TicTocStats::MinTime() {
  return static_cast<double>(min_time_ns_);
}

int32_t TicTocStats::Count() {
  return num_stats_;
}

/***********************
 * --- TicTocStats --- *
 ***********************/

TicTocBank::TicTocBank() { Reset(); }

void TicTocBank::Reset() {}

void TicTocBank::Update(const std::string &name, int64_t duration_ns) {}

/******************
 * --- TicToc --- *
 ******************/

TicToc::TicToc(const std::string &name) : name_(name) {
  Tic();
}

void TicToc::Tic() {
  start_ = chrono::high_resolution_clock::now();
}

void TicToc::Toc() {
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<int64_t, std::nano> duration = end - start_;
}

} // namespace timing
