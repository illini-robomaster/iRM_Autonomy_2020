import unittest

import cc_example_class
import cc_example_hello

class BoostPythonExampleTest(unittest.TestCase):

    def test_cc_example_hello(self):
        self.assertEqual(cc_example_hello.greet(), 'Hello World!')

    def test_cc_example_class(self):
        # test default constructor
        a = cc_example_class.WrappedClass()
        # test constructing with int
        b = cc_example_class.WrappedClass(10)
        # test constructing with string
        c = cc_example_class.WrappedClass('c')
        # test constructing with (int, string)
        d = cc_example_class.WrappedClass(20, 'd')
        # test + operator
        e = a + b + c + d
        self.assertEqual(e.get_value(), 30)
        self.assertEqual(e.name, 'cd')
        # test += operator
        e += c + d
        self.assertEqual(e.get_value(), 50)
        self.assertEqual(e.name, 'cdcd')

if __name__ == '__main__':
    unittest.main()
