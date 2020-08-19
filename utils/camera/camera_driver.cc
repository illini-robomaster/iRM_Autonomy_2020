#include "camera_driver.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>

namespace camera {

CameraBase::CameraBase(uint32_t n) {
    this->n = n;
    this->busy_index = 0;
    this->is_capturing = false;
    this->_buffer.resize(this->n);
}

void CameraBase::_start_capture(unsigned int sleep) {
    if (this->is_capturing) {
        // TODO: add legit exception handling
        std::cerr << "DONT CALL ME TWICE!!!!!\r\n";
    }
    this->is_capturing = true;
    while (true) {
        this->_buffer[this->busy_index] = this->cam_read();
        this->_index_lock.lock();
        this->busy_index = (this->busy_index + 1) % this->n;
        this->_index_lock.unlock();
        if (sleep)
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
    }
}

void CameraBase::get_img(cv::Mat &dst) {
    this->_index_lock.lock();
    uint32_t ind_to_read = (this->busy_index + this->n - 1) % this->n;
    // this can be potentially improved by taking (diminishing) tradeoff with thread safety
    // but on Jetson TX2, this implementation already runs faster than 120fps
    // which is faster than almost every camera...
    this->_buffer[ind_to_read].copyTo(dst);
    this->_index_lock.unlock();
}

void CameraBase::start() {
    // fill the buffer
    for (uint32_t i = 0; i < this->n; ++i){
        this->_buffer[i] = this->cam_read();
    }
    std::thread t(&::camera::CameraBase::_start_capture, this, 0);
    this->camera_thread_handle = t.native_handle();
    t.detach();
}

void CameraBase::stop() {
    this->_index_lock.lock();
    pthread_cancel(this->camera_thread_handle);
    this->_index_lock.unlock();
}

/* ---------------------------- SimpleCVCam ---------------------------*/

SimpleCVCam::SimpleCVCam(uint32_t n) : CameraBase(n) {
    this->cap = cv::VideoCapture(0); // default camera
}

SimpleCVCam::SimpleCVCam(uint32_t n, unsigned short device_id) : CameraBase(n) {
    this->cap = cv::VideoCapture(device_id);
}

SimpleCVCam::~SimpleCVCam() {
    this->stop();
    this->cap.release();
    cv::destroyAllWindows();
}

cv::Mat SimpleCVCam::cam_read() {
    cv::Mat frame;
    this->cap >> frame;
    return frame;
}

} // namespace camera
