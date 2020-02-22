#pragma once

#include <time.h>
#include <termios.h>
#include <string>
#include <thread>
#include <mutex>
#include <cstdint>

namespace communication {

/**
 * generic serial communication class for C++
 */
class CSerial {
 public:
  /** 
   * @brief constructor of a CSerial instance
   * @param port      directory to a serial port in the file system
   * @param baudrate  communication baudrate
   * @param data_bits data bits - [5, 6, 7, 8]
   * @param stop_bits stop bits - [1, 2]
   * @param parity    parity option - ['n', 'o', 'e', 's']
   *                      'n': none parity
   *                      'o': odd parity
   *                      'e': even parity
   *                      's': space parity
   */
  CSerial(std::string port, int baudrate, 
      int data_bits = 8, int stop_bits = 1, char parity = 'n');

  /**
   * @brief destructor of a CSerial instance
   */
  ~CSerial();

  /**
   * @brief change baudrate
   * @param baudrate
   * @return true if baudrate set successfully, otherwise false
   */
  bool set_baudrate(int baudrate);

  /**
   * @brief change parity option
   * @param parity parity option - ['n', 'o', 'e', 's']
   *          'n': none parity
   *          'o': odd parity
   *          'e': even parity
   *          's': space parity
   * @return true if parity option set successfully, otherwise false
   */
  bool set_parity(char parity);

  /**
   * @brief change data bits
   * @param data_bits number of data bits - [5, 6, 7, 8]
   * @return true if data bits set sucessfully, otherwise false
   */
  bool set_data_bits(int data_bits);

  /**
   * @brief change stop bits
   * @param stop_bits number of stop bits - [1, 2]
   * @return true if stop bits set sucessfully, otherwise false
   */
  bool set_stop_bits(int stop_bits);

  /**
   * @brief writes to the serial port and block thread
   * @param data character array
   * @param length data length
   * @return number of bytes written successfully
   */
  int write_bytes(const void* data, int length);

  /**
   * @brief reads from serial port and store data into buffer
   * @param data data buffer to store the result
   * @param length how many bytes to recieve
   * @return number of bytes read
   */
  int read_bytes(void *data, int length);

  /**
   * @brief number of bytes available in the buffer
   * @return number of bytes availble in the buffer
   */
  int bytes_available();

  /**
   * @brief flush both input & output buffer
   * @return none
   */
  void flush();

 private:
  int             fd_;
  struct termios  termios_options_;
  struct termios  old_termios_options_;
};

} /* namespace communication */

