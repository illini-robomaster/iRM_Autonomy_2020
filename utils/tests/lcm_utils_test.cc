#include <chrono>
#include <thread>

#include <lcmtypes/example/int32_vec3d_t.hpp>
#include <lcmtypes/example/int32_vec4d_t.hpp>
#include <lcmtypes/example/int32_array_t.hpp>

#include "utils/gtest_utils/test_base.h"
#include "utils/lcm_utils/logging.h"

class LCMUtilsTest : public TestBase {
 public:
  void LCMFileLoggerTest() {
    lcm::LCM lcm_backend;
    lcm::LCMFileLogger logger("/tmp/tmp.log");

    EXPECT_EQ(logger.Start(), 0);
    EXPECT_EQ(logger.Start(), -1);
    EXPECT_EQ(logger.Start(), -1);
    
    example::int32_vec3d_t vec3d = { 0, 1, 2 };
    example::int32_vec4d_t vec4d = { 0, 1, 2, 3 };
    example::int32_array_t arr;

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

    EXPECT_EQ(logger.Stop(), 0);
    EXPECT_EQ(logger.Stop(), -1);
    EXPECT_EQ(logger.Stop(), -1);
  }

  void LCMDecodeTest() {
    lcm::LogFile log("/tmp/tmp.log", "r");
    example::int32_vec3d_t vec3d;
    example::int32_vec4d_t vec4d;
    example::int32_array_t arr;

    while (1) {
      const lcm::LogEvent *event = log.readNextEvent();
      if (!event) {
        break;
      }

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
  }

};

TEST_FM(LCMUtilsTest, LCMFileLoggerTest);
TEST_FM(LCMUtilsTest, LCMDecodeTest);

