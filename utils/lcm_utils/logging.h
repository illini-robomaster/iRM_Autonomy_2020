#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <lcm/lcm-cpp.hpp>

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

  bool is_logging_;
  std::unique_ptr<std::thread> thread_ptr_;
  std::unique_ptr<lcm::LogFile> logfile_ptr_;
  std::mutex lock_;
};

} // namespace lcm
