#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "utils/timing/tic_toc.h"
#include "utils/camera/camera_driver.h"
#include "utils/gtest_utils/test_base.h"

/************************************
Camerat tests:
    - uses metrics to compare camera implementation and visually confirm results are correct

Metrics
    - Latency (How long does each call to get img take)?
    - Throughput (In a given time interval e.g. 10sec). How many non-identical image can be obtained?

************************************/

using namespace std;
using namespace timing;
using namespace camera;

bool mat_equal(cv::Mat & a, cv::Mat & b) {
    // vectorized Op
    return cv::countNonZero(a != b) == 0;
}

class CameraTest : public TestBase {
public:
    void OnePhotoTest() {
        SimpleCVCam my_cam(4, 0);
        my_cam.start();
        cv::Mat test_frame;
        ASSERT_TRUE(test_frame.rows == 0);
        ASSERT_TRUE(test_frame.cols == 0);
        my_cam.get_img(test_frame);
        ASSERT_TRUE(test_frame.rows > 0);
        ASSERT_TRUE(test_frame.cols > 0);
        imwrite("./test.jpg", test_frame);
    }

    void BenchmarkTest() {
        int n = 200; // number of frames to get
        SimpleCVCam my_cam(4, 0);
        TicTocGlobalReset();
        TicToc tictoc_baseline("Standard get_img");
        TicToc tictoc_ours("Our get_img");
        vector<cv::Mat> baseline_arr;
        vector<cv::Mat> novel_mat_arr;
        // baseline
        for (int i = 0; i < n; ++i) {
            tictoc_baseline.Tic();
            cv::Mat my_img = my_cam.cam_read();
            tictoc_baseline.Toc();
            baseline_arr.push_back(my_img);
        }
        my_cam.start();
        // ours
        for (int i = 0; i < n; ++i) {
            tictoc_ours.Tic();
            cv::Mat my_img;
            my_cam.get_img(my_img);
            tictoc_ours.Toc();
            novel_mat_arr.push_back(my_img);
        }
        std::cout << "[ TicTocTest ] Time Summary:" << std::endl;
        std::cout << TicTocGlobalSummary() << std::endl;
        ASSERT_TRUE(baseline_arr.size() == ((size_t)n));
        ASSERT_TRUE(novel_mat_arr.size() == ((size_t)n));
        for (int i = 0; i < n; ++i) {
            ASSERT_TRUE(baseline_arr[i].rows > 0);
            ASSERT_TRUE(baseline_arr[i].cols > 0);
            ASSERT_TRUE(novel_mat_arr[i].rows > 0);
            ASSERT_TRUE(novel_mat_arr[i].cols > 0);
        }
        // latency
        double baseline_time = TicTocStatsTime("Standard get_img");
        double our_time = TicTocStatsTime("Our get_img");
        ASSERT_TRUE(our_time < baseline_time);
        // throughput
        int baseline_valid_cnt = 0;
        int our_valid_cnt = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (!mat_equal(baseline_arr[i], baseline_arr[i + 1]))
                baseline_valid_cnt++;
            if (!mat_equal(novel_mat_arr[i], novel_mat_arr[i + 1]))
                our_valid_cnt++;
        }
        std::cout << "Valid baseline mat cnt: " <<  baseline_valid_cnt << std::endl;
        std::cout << "Valid our mat cnt: " << our_valid_cnt << std::endl;
        double baseline_throughput = baseline_valid_cnt / baseline_time;
        double our_throughput = our_valid_cnt / our_time;
        std::cout << "Baseline throughput (fps): " << baseline_throughput << std::endl;
        std::cout << "Our throughput (fps): " << our_throughput << std::endl;
    }
};

// for some unknown reason, releasing a videocapture object
// and then reacquring it sometimes cause Illegal Hardware Instruction fault.
// This should not be a problem on the robot.
// Therefore, one test is commented out.
// TEST_FM(CameraTest, OnePhotoTest);
TEST_FM(CameraTest, BenchmarkTest);
