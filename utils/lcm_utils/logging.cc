#include "utils/lcm_utils/logging.h"
#include "utils/timing/tic_toc.h"

namespace lcm {

LCMFileLogger::LCMFileLogger(const std::string &filename, const std::string &lcm_url)
    : lcm_(lcm_url), filename_(filename), is_logging_(false), subscription_(nullptr) {}

LCMFileLogger::~LCMFileLogger() { Stop(); }

void LCMFileLogger::EventLoop() {
  while (true) {
    // check for logging status
    {
      std::lock_guard<std::mutex> lg(lock_);
      if (!is_logging_) {
        return;
      }
    }
    // main logging event loop
    if (lcm_.handleTimeout(1) < 0) {  // something is wrong here
      std::lock_guard<std::mutex> lg(lock_);
      is_logging_ = false;
    }
  }
}

void LCMFileLogger::MessageHandler(const ReceiveBuffer *rbuf, const std::string &channel) {
  lcm::LogEvent event;
  event.timestamp = rbuf->recv_utime;
  event.channel = channel;
  event.data = rbuf->data;
  event.datalen = rbuf->data_size;
  logfile_ptr_->writeEvent(&event);
}

int LCMFileLogger::Start() {
  {
    std::lock_guard<std::mutex> lg(lock_);
    if (is_logging_)
      return -1;
    is_logging_ = true;
  }
  subscription_ = lcm_.subscribe(".*", &LCMFileLogger::MessageHandler, this);
  logfile_ptr_ = std::make_unique<lcm::LogFile>(filename_, "w");
  thread_ptr_ = std::make_unique<std::thread>(&LCMFileLogger::EventLoop, this);

  return 0;
}

int LCMFileLogger::Stop() {
  {
    std::lock_guard<std::mutex> lg(lock_);
    if (!is_logging_) {
      return -1;
    }
    is_logging_ = false;
  }
  // wait for event loop to exit
  thread_ptr_->join();
  // explicitly release allocated resources
  logfile_ptr_.reset();
  thread_ptr_.reset();
  lcm_.unsubscribe(subscription_);
  subscription_ = nullptr;

  return 0;
}

} // namespace lcm
