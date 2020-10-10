#include <chrono>
#include <thread>
#include <unordered_map>

#include <lcmtypes/example/int32_vec3d_t.hpp>
#include <lcmtypes/example/int32_vec4d_t.hpp>
#include <lcmtypes/example/int32_array_t.hpp>

#include "utils/gtest_utils/test_base.h"
#include "utils/lcm_utils/logging.h"

#define TEMP_LOG_PRODUCT "/tmp/lcm_utils_test.lcmlog"

class LCMUtilsTest : public TestBase {
 protected:
  /**
   * @brief example usage of LCMFileLogger and expected behaviors
   */
  void SetUp() override {
    lcm::LCM lcm_backend;
    lcm::LCMFileLogger logger(TEMP_LOG_PRODUCT);
    // test: consecutive start should fail
    EXPECT_EQ(logger.Start(), 0);
    EXPECT_EQ(logger.Start(), -1);
    EXPECT_EQ(logger.Start(), -1);
    // define some example data structs
    example::int32_vec3d_t vec3d = { 0, 1, 2 };
    example::int32_vec4d_t vec4d = { 0, 1, 2, 3 };
    example::int32_array_t arr;
    // test: burst publish should not mess up the logger
    EXPECT_EQ(lcm_backend.publish("INT32_VEC3D", &vec3d), 0);
    EXPECT_EQ(lcm_backend.publish("INT32_VEC3D", &vec3d), 0);
    EXPECT_EQ(lcm_backend.publish("INT32_VEC3D", &vec3d), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_EQ(lcm_backend.publish("INT32_VEC4D", &vec4d), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    arr.size = 100;
    arr.data.resize(arr.size);
    for (int i = 0; i < arr.size; ++i) {
      arr.data[i] = i;
    }
    EXPECT_EQ(lcm_backend.publish("INT32_ARRAY", &arr), 0);
    arr.size = 20;
    arr.data.resize(arr.size);
    for (int i = 0; i < arr.size; ++i) {
      arr.data[i] = i;
    }
    EXPECT_EQ(lcm_backend.publish("INT32_ARRAY", &arr), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // test: consecutive stops shoule fail
    EXPECT_EQ(logger.Stop(), 0);
    EXPECT_EQ(logger.Stop(), -1);
    EXPECT_EQ(logger.Stop(), -1);
    // test: restart logger and append to existing log after stopped
    EXPECT_EQ(logger.Start(false), 0);
    EXPECT_EQ(logger.Start(false), -1);
    EXPECT_EQ(lcm_backend.publish("INT32_VEC3D", &vec3d), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // test: logger goes out of scope should automatically stop logging
  }

  void LCMDecodeTest() {
    lcm::LogFile log(TEMP_LOG_PRODUCT, "r");
    example::int32_vec3d_t vec3d;
    example::int32_vec4d_t vec4d;
    example::int32_array_t arr;
    // keep track of number of messages per channel
    std::unordered_map<std::string, int> count;

    while (1) {
      const lcm::LogEvent *event = log.readNextEvent();
      if (!event) { // reach end of log file
        break;
      }
      // update message per channel count
      if (count.find(event->channel) == count.end()) {
        count[event->channel] = 0;
      }
      ++count[event->channel];
      // test: decode log file
      std::cout << "[ Decoding ] " << event->channel << " at " << event->timestamp << std::endl;
      if (event->channel == "INT32_VEC3D") {
        vec3d.decode(event->data, 0, event->datalen);
        EXPECT_EQ(vec3d.x, 0);
        EXPECT_EQ(vec3d.y, 1);
        EXPECT_EQ(vec3d.z, 2);
      }
      else if (event->channel == "INT32_VEC4D") {
        vec4d.decode(event->data, 0, event->datalen);
        EXPECT_EQ(vec4d.x, 0);
        EXPECT_EQ(vec4d.y, 1);
        EXPECT_EQ(vec4d.z, 2);
        EXPECT_EQ(vec4d.w, 3);
      }
      else if (event->channel == "INT32_ARRAY") {
        arr.decode(event->data, 0, event->datalen);
        EXPECT_EQ(arr.size, arr.data.size());
        for (int i = 0; i < arr.size; ++i) {
          EXPECT_EQ(arr.data[i], i);
        }
      }
      else {
        FAIL(); // should not get here
      }
      std::cout << "[ Succeed  ] " << event->channel << " of size "<< event->datalen<< std::endl;
    }
    // test: expected number of messages per channel
    EXPECT_EQ(count["INT32_VEC3D"], 4);
    EXPECT_EQ(count["INT32_VEC4D"], 1);
    EXPECT_EQ(count["INT32_ARRAY"], 2);
  }
};

TEST_FM(LCMUtilsTest, LCMDecodeTest);

