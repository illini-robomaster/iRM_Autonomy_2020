#include <iomanip>
#include <map>
#include <sstream>

#include "utils/timing/tic_toc.h"

using namespace std;

namespace timing {

/**************************
 * --- Global Context --- *
 **************************/

static TicTocBank global_tic_toc_bank;

void TicTocGlobalUpdate(const std::string &name, int64_t duration_ns) {
  global_tic_toc_bank.Update(name, duration_ns);
}

void TicTocGlobalReset() {
  global_tic_toc_bank.Reset();
}

std::string TicTocGlobalSummary() {
  std::unordered_map<std::string, TicTocStats> raw_map = global_tic_toc_bank.GetSummary();
  // sort according to keys + find maximum channel name length
  std::map<std::string, TicTocStats> ordered_map;
  size_t max_name_length = std::string("Channel Name").length() + 1;
  for (auto it = raw_map.begin(); it != raw_map.end(); ++it) {
    ordered_map[it->first] = it->second;
    max_name_length = std::max(max_name_length, it->first.length() + 1);
  }
  // generate summary string
  std::stringstream ss;
  ss << std::left;
  ss << "| " << std::setw(max_name_length) << "Channel Name";
  ss << "| " << std::setw(20) << "Count";
  ss << "| " << std::setw(20) << "Total Time (s)";
  ss << "| " << std::setw(20) << "Min Time (s)";
  ss << "| " << std::setw(20) << "Average Time (s)";
  ss << "| " << std::setw(20) << "Max Time (s)";
  ss << "|" << std::endl;
  for (auto it = ordered_map.begin(); it != ordered_map.end(); ++it) {
    ss << "| " << std::setw(max_name_length) << it->first;
    ss << "| " << std::setw(20) << it->second.Count();
    ss << "| " << std::setw(20) << it->second.TotalTime();
    ss << "| " << std::setw(20) << it->second.MinTime();
    ss << "| " << std::setw(20) << it->second.AverageTime();
    ss << "| " << std::setw(20) << it->second.MaxTime() << "|" << std::endl;
  }

  return ss.str();
}

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
  return static_cast<double>(total_time_ns_) * kSecPerNanoSec;
}

double TicTocStats::AverageTime() {
  return TotalTime() / num_stats_;
}

double TicTocStats::MaxTime() {
  return static_cast<double>(max_time_ns_) * kSecPerNanoSec;
}

double TicTocStats::MinTime() {
  return static_cast<double>(min_time_ns_) * kSecPerNanoSec;
}

int32_t TicTocStats::Count() {
  return num_stats_;
}

/**********************
 * --- TicTocBank --- *
 **********************/

TicTocBank::TicTocBank() { Reset(); }

void TicTocBank::Reset() {
  channel_map_.clear();
}

void TicTocBank::Update(const std::string &name, int64_t duration_ns) {
  if (channel_map_.find(name) == channel_map_.end()) {
    channel_map_[name] = TicTocStats();
  }
  channel_map_[name].Update(duration_ns);
}

std::unordered_map<std::string, TicTocStats> TicTocBank::GetSummary() {
  return channel_map_;
}

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
  TicTocGlobalUpdate(name_, duration.count());
}

/************************
 * --- ScopedTicToc --- *
 ************************/

ScopedTicToc::ScopedTicToc(const std::string &name) : name_(name) {
  start_ = chrono::high_resolution_clock::now();
}

ScopedTicToc::~ScopedTicToc() {
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<int64_t, std::nano> duration = end - start_;
  TicTocGlobalUpdate(name_, duration.count());
}

} // namespace timing
