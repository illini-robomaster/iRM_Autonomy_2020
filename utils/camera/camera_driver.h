#pragma once

#include <mutex>
#include <vector>
#include <thread>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace std;

namespace camera {
    /**
     * A base class for generalized camera level processing
     */
    class CameraBase {
    public:
        /**
         * @brief constructor for CameraBase class
         *
         * @param n     size of circular frame buffer
         */
        CameraBase(uint32_t n);

        /**
         * @brief read an image and copy into src Mat
         *
         * @param dst   destination place holder for the stored image
         * @return none
         */
        void get_img(cv::Mat &dst);

        /**
         * @brief start the camera in a seperate thread
         *
         * @return none
         */
        void start();
        
        /**
         * @brief terminates the (already started) separate thread
         */
        void stop();

        /**
         * @brief to be implemented according to the specs of a specific camera
         *
         * @return Mat object read directly from the camera
         */
        virtual cv::Mat cam_read() = 0;

        /**
         * @brief determine if the camera is dead or alive. Should only be of use for video feed
         *
         * @return alive flag
         */
        inline bool is_alive(void) { return this->is_capturing; }
    protected:
        /**
         * @brief Invoke a separate thread to get images
         *
         * @param sleep     time (in ms) to sleep between each capture. Default to zero
         * @return none
         */
        void _start_capture(unsigned int sleep = 0);

        bool is_capturing; // maybe use a lock?
        uint32_t n;
        // index indicating frames of interest
        // increment by each capturing loop
        uint32_t busy_index;
        std::vector<cv::Mat> _buffer;
        mutex _index_lock;
        pthread_t camera_thread_handle;
    };

    /**
     * @brief a generalized opencv camera class
     */
    class SimpleCVCam: public CameraBase {
    public:
        /**
         * @brief constructor for SimpleCVCam class
         *
         * @param n         size of circular frame buffer
         */
        SimpleCVCam(uint32_t n);

        /**
         * @brief constructor for SimpleCVCam
         *
         * @param n         size of the circular frame buffer
         * @param device_id device id as in camera index [default to 0]
         */
        SimpleCVCam(uint32_t n, unsigned short device_id);

        ~SimpleCVCam();

        /**
         * @brief   virtual implementation of how to read images
         *
         * @return  Mat object read directly from the camera
         */
        cv::Mat cam_read();
    protected:
        cv::VideoCapture cap;
};
}
