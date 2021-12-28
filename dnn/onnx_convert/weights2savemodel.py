import tensorflow as tf
from dnn.model.yolov3_tiny import YoloV3Tiny
from dnn.parameters import PARAM
from dnn.utils.mem import tf_set_memory_growth

def main():
    # Get params
    size = PARAM['size']
    num_cls = PARAM['class_number']
    model = YoloV3Tiny(size, classes=num_cls, training=False)

    # load model
    import os
    checkpoint_dir = './checkpoints/yolov3_train_100.tf'
    model.load_weights(checkpoint_dir)

    tf.saved_model.save(model, './save_onnx/save_model_format')


if __name__ == '__main__':
    # set memory growth
    tf_set_memory_growth()
    main()

