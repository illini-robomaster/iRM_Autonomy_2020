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


    import tf2onnx
    import onnxruntime as rt

    spec = (tf.TensorSpec((1, 416, 416, 3), tf.float32, name="input"),)
    output_path = './save_onnx/irm_' + model.name + ".onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=12, output_path=output_path)

    print(f'Model saved at: {output_path}')
    import onnx
    onnx.checker.check_model(output_path)


if __name__ == '__main__':
    # set memory growth
    tf_set_memory_growth()
    main()

