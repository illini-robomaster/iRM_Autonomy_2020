#include <iostream>

#include "utils/timing/tic_toc.h"
#include "utils/gtest_utils/test_base.h"

using namespace timing;

class TimingTest : public TestBase {
 public:
  void TicTocTest() {
    TicTocGlobalReset();
    TicToc tictoc_out("Outer Loop");
    TicToc tictoc_in("Inner Loop");
    // first run
    tictoc_out.Tic();
    for (double i = 0; i < 1000; ++i) {
      tictoc_in.Tic();
      for (double j = 0; j < 1000; ++j);
      tictoc_in.Toc();
    }
    tictoc_out.Toc();
    // second run
    tictoc_out.Tic();
    for (double i = 0; i < 1000; ++i) {
      tictoc_in.Tic();
      for (double j = 0; j < 1000; ++j);
      tictoc_in.Toc();
    }
    tictoc_out.Toc();
    // third run
    tictoc_out.Tic();
    for (double i = 0; i < 1000; ++i) {
      tictoc_in.Tic();
      for (double j = 0; j < 1000; ++j);
      tictoc_in.Toc();
    }
    tictoc_out.Toc();
    std::cout << "[ TicTocTest ] Time Summary:" << std::endl;
    std::cout << TicTocGlobalSummary() << std::endl;
  }

  void TicTocScopeTest() {
    TicTocGlobalReset();

    {
      TIC_TOC_SCOPE(TicTocScopeTest);
      for (double i = 0; i < 1000000; ++i);
    }

    std::cout << "[ TicTocScopeTest ] Time Summary:" << std::endl;
    std::cout << TicTocGlobalSummary() << std::endl;
  }

  void TicTocFunctionTest() {
    TicTocGlobalReset();

    TimeConsumingFunction();
    TimeConsumingFunction();
    TimeConsumingFunction();
    TimeConsumingFunction();
    TimeConsumingFunction();

    std::cout << "[ TicTocFunctionTest ] Time Summary:" << std::endl;
    std::cout << TicTocGlobalSummary() << std::endl;
  }

 private:
  void TimeConsumingFunction() {
    TIC_TOC_FUNCTION();
    for (double i = 0; i < 1000000; ++i);
  }
};

TEST_FM(TimingTest, TicTocTest)
TEST_FM(TimingTest, TicTocScopeTest)
TEST_FM(TimingTest, TicTocFunctionTest)
