#include "utils/gtest_utils/test_base.h"
#include "utils/lcm_utils/logging.h"
#include "utils/timing/tic_toc.h"

class LCMUtilsTest : public TestBase {
 public:
  void LCMLoggingAPITest() {
    lcm::LCMFileLogger logger("/tmp/tmp.log");
    EXPECT_EQ(logger.Start(), 0);
    EXPECT_EQ(logger.Start(), -1);
    EXPECT_EQ(logger.Stop(), 0);
    EXPECT_EQ(logger.Stop(), -1);
  }
};

TEST_FM(LCMUtilsTest, LCMLoggingAPITest);
