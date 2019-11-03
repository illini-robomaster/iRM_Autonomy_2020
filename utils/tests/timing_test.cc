#include "utils/timing/tic_toc.h"
#include "utils/gtest_utils/test_base.h"

using namespace timing;

class TimingTest : public TestBase {
 public:
  void TicTocTest() {
    TicToc tictoc("Raw TicToc");
    for (double i = 0; i < 100000; ++i);
    tictoc.Toc();
  }
};

TEST_FM(TimingTest, TicTocTest)
