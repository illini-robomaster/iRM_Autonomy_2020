#include "serial.h"
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

using namespace std;

namespace communication {

CSerial::CSerial(string port, int baudrate, int data_bits, 
    int stop_bits, char parity) {
  fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd_ == -1)
    cout << "serial port open failed!" << endl;
  if (tcgetattr(fd_, &old_termios_options_))
    cout << "get serial attributes error!" << endl;
  termios_options_ = old_termios_options_;
  termios_options_.c_cflag |= (CLOCAL | CREAD);
  termios_options_.c_cflag &= (~CSIZE & ~CRTSCTS);
  if (!set_baudrate(baudrate))
    cout << "baudrate " << baudrate << " does not exist!" << endl;
  if (!set_data_bits(data_bits))
    cout << "failed to set " << data_bits << " data bits!" << endl;
  if (!set_stop_bits(stop_bits))
    cout << "failed to set " << stop_bits << " stop bits!" << endl;
  if (!set_parity(parity))
    cout << parity << " parity does not exist!" << endl;
  termios_options_.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
  termios_options_.c_oflag &= ~OPOST;
  termios_options_.c_iflag = IGNBRK;
  termios_options_.c_cc[VTIME] = 1;
  termios_options_.c_cc[VMIN] = 0;
  if (tcsetattr(fd_, TCSANOW, &termios_options_))
    cout << "set attribute error!" << endl;
}

CSerial::~CSerial() {
  tcsetattr(fd_, TCSANOW, &old_termios_options_);
  close(fd_);
}

bool CSerial::set_baudrate(int baudrate) {
  const int speed_arr[] = {B115200, B19200, B9600, B4800, B2400, B1200, B300};
  const int name_arr[] = {115200, 19200, 9600, 4800, 2400, 1200, 300};

  for (size_t i = 0; i < 7; i++)
    if (baudrate == name_arr[i]) {
      cfsetispeed(&termios_options_, speed_arr[i]);
      cfsetospeed(&termios_options_, speed_arr[i]);
      return true;
    }

  return false;
}

bool CSerial::set_data_bits(int data_bits) {
  switch (data_bits) {
    case 5: termios_options_.c_cflag |= CS5;
            return true;
    case 6: termios_options_.c_cflag |= CS6;
            return true;
    case 7: termios_options_.c_cflag |= CS7;
            return true;
    case 8: termios_options_.c_cflag |= CS8;
            return true;
    default: return false;
  }
}

bool CSerial::set_stop_bits(int stop_bits) {
  switch(stop_bits) {
    case 1: termios_options_.c_cflag &= ~CSTOPB;
            return true;
    case 2: termios_options_.c_cflag |= CSTOPB;
            return true;
    default: return false;
  }
}

bool CSerial::set_parity(char parity) {
  switch (parity) {
    case 'n':
    case 'N': termios_options_.c_cflag &= ~(PARENB | PARODD);
              termios_options_.c_iflag &= ~INPCK;
              return true;
    case 'o':
    case 'O': termios_options_.c_cflag |= (PARODD | PARENB);
              termios_options_.c_iflag |= INPCK;
              return true;
    case 'e':
    case 'E': termios_options_.c_cflag |= PARENB;
              termios_options_.c_cflag &= ~PARODD;
              termios_options_.c_iflag |= INPCK;
              return true;
    case 's':
    case 'S': termios_options_.c_cflag &= ~PARENB;
              termios_options_.c_cflag &= ~CSTOPB;
              return true;
    default: return false;
  }
}

int CSerial::write_bytes(const void *data, int length) {
  return write(fd_, data, length);
}

int CSerial::read_bytes(void *data, int length) {
  if (!bytes_available())
    return 0;
  return read(fd_, data, length);
}

int CSerial::bytes_available() {
  int bytes;
  ioctl(fd_, FIONREAD, &bytes);
  return bytes;
}

void CSerial::flush() {
  tcflush(fd_, TCIOFLUSH);
  return;
}

} /* namespace communication */
