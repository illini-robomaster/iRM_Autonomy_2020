#include <iostream>
#include <thread>

#include "utils/timing/tic_toc.h"
#include "utils/gtest_utils/test_base.h"

using namespace timing;

// prevent loop body from being opimized out
volatile thread_local double dummy_variable;

class TimingTest : public TestBase {
 public:
   /**
    * @brief example usage of basic TicToc profiler
    */
  void TicTocTest() {
    TicTocGlobalReset();
    TicToc tictoc_out("Outer Loop");
    TicToc tictoc_in("Inner Loop");
    // first run
    tictoc_out.Tic();
    for (double i = 0; i < 1000; ++i) {
      tictoc_in.Tic();
      for (double j = 0; j < 1000; ++j) {
        dummy_variable = i + j;
      }
      tictoc_in.Toc();
    }
    tictoc_out.Toc();
    // second run
    tictoc_out.Tic();
    for (double i = 0; i < 1000; ++i) {
      tictoc_in.Tic();
      for (double j = 0; j < 1000; ++j) {
        dummy_variable = i + j;
      }
      tictoc_in.Toc();
    }
    tictoc_out.Toc();
    // Tic / Toc macro (not reusable with the same name within the same scope)
    TIC(TOTAL_LOOP);
    for (double i = 0; i < 1000; ++i) {
      for (double j = 0; j < 1000; ++j) {
        dummy_variable = i + j;
      }
    }
    TOC(TOTAL_LOOP);
    std::cout << "[ TicTocTest ] Time Summary:" << std::endl;
    std::cout << TicTocGlobalSummary() << std::endl;
  }

  /**
   * @brief example usage of TIC_TOC_SOPE
   */
  void TicTocScopeTest() {
    TicTocGlobalReset();

    {
      TIC_TOC_SCOPE(TicTocScopeTest);
      for (double i = 0; i < 1000000; ++i) {
        dummy_variable = i;
      }
    }

    std::cout << "[ TicTocScopeTest ] Time Summary:" << std::endl;
    std::cout << TicTocGlobalSummary() << std::endl;
  }

  /**
   * @brief example usage of TIC_TOC_FUNCTION
   */
  void TicTocFunctionTest() {
    TicTocGlobalReset();

    TimeConsumingFunction();
    TimeConsumingFunction();
    TimeConsumingFunction();
    TimeConsumingFunction();
    TimeConsumingFunction();

    // test: multi thread
    std::thread t1(&TimingTest::TimeConsumingFunction, this);
    std::thread t2(&TimingTest::TimeConsumingFunction, this);

    t1.detach();
    t2.join();

    std::cout << "[ TicTocFunctionTest ] Time Summary:" << std::endl;
    std::cout << TicTocGlobalSummary() << std::endl;
  }

 private:
  void TimeConsumingFunction() {
    TIC_TOC_FUNCTION();
    for (double i = 0; i < 1000000; ++i) {
      dummy_variable = i;
    }
  }
};

TEST_FM(TimingTest, TicTocTest)
TEST_FM(TimingTest, TicTocScopeTest)
TEST_FM(TimingTest, TicTocFunctionTest)
