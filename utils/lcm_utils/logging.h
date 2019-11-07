#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <lcm/lcm-cpp.hpp>

typedef std::chrono::duration<int64_t, std::micro> micro_sec_t;

namespace lcm {

class LCMFileLogger {
 public:
  LCMFileLogger(const std::string &filename, const std::string &lcm_url="");
  ~LCMFileLogger();

  int Start();
  int Stop();

 private:
  void EventLoop();
  void MessageHandler(const lcm::ReceiveBuffer *rbuf, const std::string &channel);

  std::string filename_;

  lcm::LCM lcm_;
  lcm::Subscription *subscription_;

  std::chrono::steady_clock::time_point start_time_;
  std::future<void> stop_signal_;
  std::unique_ptr<std::promise<void>> promise_start_;
  std::unique_ptr<std::thread> thread_ptr_;
  std::unique_ptr<lcm::LogFile> logfile_ptr_;
};

} // namespace lcm
