#include <functional>

#include "utils/lcm_utils/logging.h"
#include "utils/timing/tic_toc.h"

namespace lcm {

LCMFileLogger::LCMFileLogger(const std::string &filename, const std::string &lcm_url)
    : lcm_(lcm_url), filename_(filename), subscription_(nullptr) {}

LCMFileLogger::~LCMFileLogger() { Stop(); }

void LCMFileLogger::EventLoop() {
  while (stop_signal_.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
    lcm_.handleTimeout(0);
  }
}

void LCMFileLogger::MessageHandler(const ReceiveBuffer *rbuf, const std::string &channel) {
  auto now = std::chrono::steady_clock::now();
  micro_sec_t timestamp = std::chrono::duration_cast<micro_sec_t>(now - start_time_);
  lcm::LogEvent event;
  event.timestamp = timestamp.count();
  event.channel = channel;
  event.data = rbuf->data;
  event.datalen = rbuf->data_size;
  logfile_ptr_->writeEvent(&event);
}

int LCMFileLogger::Start(bool overwrite) {
  if (promise_start_) {
    return -1;
  }
  promise_start_ = std::make_unique<std::promise<void>>();
  stop_signal_ = promise_start_->get_future();
  subscription_ = lcm_.subscribe(".*", &LCMFileLogger::MessageHandler, this);
  logfile_ptr_ = std::make_unique<lcm::LogFile>(filename_, overwrite ? "w" : "a");
  start_time_ = std::chrono::steady_clock::now();
  thread_ptr_ = std::make_unique<std::thread>(&LCMFileLogger::EventLoop, this);

  return 0;
}

int LCMFileLogger::Stop() {
  if (!promise_start_) {
    return -1;
  }
  // send terminate signal to event loop
  promise_start_->set_value();
  // wait for event loop to exit
  thread_ptr_->join();
  // explicitly release allocated resources
  promise_start_.reset();
  logfile_ptr_.reset();
  thread_ptr_.reset();
  lcm_.unsubscribe(subscription_);
  subscription_ = nullptr;

  return 0;
}

} // namespace lcm
