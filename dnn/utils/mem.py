import tensorflow as tf


def tf_set_memory_growth():
    """set all your GPU to add more memory when necessary for TensorFlow
    """
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
