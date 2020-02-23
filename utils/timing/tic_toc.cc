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
  tictoc_t tictocs = global_tic_toc_bank.GetLCM();
  // sort according to keys + find maximum channel name length
  std::map<std::string, TicTocStats> ordered_map;
  size_t max_name_length = std::string("Channel Name").length() + 1;
  for (auto &tictoc: tictocs.tictoc_channels) {
    ordered_map[tictoc.name] = tictoc.tictoc_stats;
    max_name_length = std::max(max_name_length, tictoc.name.length() + 1);
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

double TicTocStatsTime(std::string query) {
  tictoc_t tictocs = global_tic_toc_bank.GetLCM();
  // sort according to keys + find maximum channel name length
  for (tictoc_channel_t &tictoc: tictocs.tictoc_channels) {
    if (tictoc.name == query)
      return ((TicTocStats)(tictoc.tictoc_stats)).TotalTime();
  }
  return -1; // not found
}

/***********************
 * --- TicTocStats --- *
 ***********************/

TicTocStats::TicTocStats() { Reset(); }

TicTocStats::TicTocStats(tictoc_stats_t &stats) {
  num_ticks = stats.num_ticks;
  min_time_ns = stats.min_time_ns;
  max_time_ns = stats.max_time_ns;
  total_time_ns = stats.total_time_ns;
}

void TicTocStats::Reset() {
  num_ticks = 0;
  min_time_ns = std::numeric_limits<int64_t>::max();
  max_time_ns = std::numeric_limits<int64_t>::min();
  total_time_ns = 0;
}

void TicTocStats::Update(int64_t duration_ns) {
  ++num_ticks;
  min_time_ns = std::min(min_time_ns, duration_ns);
  max_time_ns = std::max(max_time_ns, duration_ns);
  total_time_ns += duration_ns;
}

double TicTocStats::TotalTime() {
  return static_cast<double>(total_time_ns) * kSecPerNanoSec;
}

double TicTocStats::AverageTime() {
  return TotalTime() / num_ticks;
}

double TicTocStats::MaxTime() {
  return static_cast<double>(max_time_ns) * kSecPerNanoSec;
}

double TicTocStats::MinTime() {
  return static_cast<double>(min_time_ns) * kSecPerNanoSec;
}

int32_t TicTocStats::Count() {
  return num_ticks;
}

/**********************
 * --- TicTocBank --- *
 **********************/

TicTocBank::TicTocBank() { Reset(); }

void TicTocBank::Reset() {
  std::lock_guard<std::mutex> guard(lock_);
  channel_map_.clear();
}

void TicTocBank::Update(const std::string &name, int64_t duration_ns) {
  std::lock_guard<std::mutex> guard(lock_);
  if (channel_map_.find(name) == channel_map_.end()) {
    channel_map_[name] = TicTocStats();
  }
  channel_map_[name].Update(duration_ns);
}

tictoc_t TicTocBank::GetLCM() {
  std::lock_guard<std::mutex> guard(lock_);
  tictoc_t ret;
  ret.num_channels = channel_map_.size();

  for (auto it = channel_map_.begin(); it != channel_map_.end(); ++it) {
    ret.tictoc_channels.push_back({ it->first, it->second });
  }

  return ret;
}

/******************
 * --- TicToc --- *
 ******************/

TicToc::TicToc(const std::string &name) : name_(name) {
  Tic();
}

void TicToc::Tic() {
  start_ = chrono::steady_clock::now();
}

void TicToc::Toc() {
  auto end = chrono::steady_clock::now();
  nano_sec_t duration = chrono::duration_cast<nano_sec_t>(end - start_);
  TicTocGlobalUpdate(name_, duration.count());
}

/************************
 * --- ScopedTicToc --- *
 ************************/

ScopedTicToc::ScopedTicToc(const std::string &name) : name_(name) {
  start_ = chrono::steady_clock::now();
}

ScopedTicToc::~ScopedTicToc() {
  auto end = chrono::steady_clock::now();
  nano_sec_t duration = chrono::duration_cast<nano_sec_t>(end - start_);
  TicTocGlobalUpdate(name_, duration.count());
}

} // namespace timing
