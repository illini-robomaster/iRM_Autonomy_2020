from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf

input_saved_model_dir = './save_onnx/save_model_format'
output_saved_model_dir = './save_onnx/trt'
params = tf.experimental.tensorrt.ConversionParams(
    precision_mode='FP16')
converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir=input_saved_model_dir, conversion_params=params)
converter.convert()
converter.save(output_saved_model_dir)
