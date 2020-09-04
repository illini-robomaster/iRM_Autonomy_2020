"""
This file demonstrate how parameters can be managed using ParamDict util

run `python params.py -f params.py` can show how parameters can be loaded from file
"""

import argparse
import tensorflow as tf

from pprint import pprint

from dnn.utils.params import ParamDict as o

# some param blocks could share parameters
shared = o(
    image_size = (100, 100),
    batch_size = 20,
)

# root params
PARAMS = o(
    # can store number
    my_num = 10,
    # can store string
    my_str = 'abc',
    # can store array
    my_array = [3, 2, 1],
    # can store class constructor
    my_class = tf.optimizers.Adam,
    # can store function
    my_func = lambda x: x**2,
    # can store object
    my_obj = tf.optimizers.Adam(learning_rate=0.1, epsilon=1e-3),
    # can store nested param blocks
    my_params1 = o(
        my_num = 100,
    ),
    my_params2 = o(
        my_num = 200,
    ),
)

# we could inject shared parameters into different levels of a param block
def resolve_dependency(params):
    params.update(shared)
    params.my_params1.update(shared)
    params.my_params2.update(shared)

resolve_dependency(PARAMS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("param block example")
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='path to a parameter file')
    args = parser.parse_args()
    params = o.from_file(args.file)
    print('Parsed params:')
    pprint(params, indent=4)
