#include <string>

#include <boost/python.hpp>

using namespace std;
using namespace boost::python;

class WrappedClass {
 public:
  WrappedClass() : value_(0), name_("") {}
  WrappedClass(int value) : value_(value), name_("") {}
  WrappedClass(const std::string& name): value_(0), name_(name) {}
  WrappedClass(int value, const std::string& name) : value_(value), name_(name) {}

  int get_value() {
    return value_;
  }

  void set_value(int value) {
    value_ = value;
  }

  const std::string get_name() {
    return name_;
  }

  void set_name(const std::string& name) {
    name_ = name;
  }

  WrappedClass operator+=(const WrappedClass &other) {
    value_ += other.value_;
    name_ += other.name_;
    return *this;
  }

  WrappedClass operator+(const WrappedClass &other) {
    return WrappedClass(value_ + other.value_, name_ + other.name_);
  }

 private:
  int value_;
  std::string name_;
};

BOOST_PYTHON_MODULE(cc_example_class) {
  class_<WrappedClass>("WrappedClass")
    .def(init<int>())
    .def("get_value", &WrappedClass::get_value)
    .def("set_value", &WrappedClass::set_value)
    .add_property("name", &WrappedClass::get_name, &WrappedClass::set_name)
    .def(self + self)
    .def(self += self);
}
