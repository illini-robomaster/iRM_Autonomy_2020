#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <lcm/lcm-cpp.hpp>

// TODO(alvin): maybe move this to a more general location later?
typedef std::chrono::duration<int64_t, std::micro> micro_sec_t;

namespace lcm {

/**
 * @brief An LCM file logger that listens to all channels being published
 */
class LCMFileLogger {
 public:
   /**
    * @brief constructor
    *
    * @param filename   filename to store log product
    * @param lcm_url    lcm url, see lcm documentation for more [Default to UDP]
    */
  LCMFileLogger(const std::string &filename, const std::string &lcm_url="");

  /**
   * @brief destructor (explictly force to stop logging)
   */
  ~LCMFileLogger();

  /**
   * @brief start listening to the lcm url and logging whatever data recieved
   *
   * @param overwrite set to false to append to the existing log files
   *
   * @return 0 on success, -1 on error
   */
  int Start(bool overwrite = true);

  /**
   * @brief stop logging and free up resources
   *
   * @return 0 on succes, -1 on error
   */
  int Stop();

 private:
  /**
   * @brief main event loop that logs data to file @ maximum 1000Hz
   */
  void EventLoop();

  /**
   * @brief handle incoming messages
   *
   * @param rbuf    received data buffer
   * @param channel channel name
   */
  void MessageHandler(const lcm::ReceiveBuffer *rbuf, const std::string &channel);

  std::string filename_;

  lcm::LCM lcm_;
  lcm::Subscription *subscription_;

  std::future<void> stop_signal_;
  std::unique_ptr<std::chrono::steady_clock::time_point> start_time_;
  std::unique_ptr<std::promise<void>> promise_start_;
  std::unique_ptr<std::thread> thread_ptr_;
  std::unique_ptr<lcm::LogFile> logfile_ptr_;
};

} // namespace lcm
