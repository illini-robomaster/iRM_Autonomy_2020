import onnx
from google.protobuf.json_format import MessageToDict
import argparse

parser = argparse.ArgumentParser(description='Inspect an onnx model.')
parser.add_argument('--model_file', type=str, default='./save_onnx/model.onnx',
                    help='onnx model file to use')

args = parser.parse_args()

model = onnx.load(args.model_file)
print('Inputs:')
for _input in model.graph.input:
    print(MessageToDict(_input))
print('=================================')
print('Outputs:')
for output in model.graph.output:
    print(MessageToDict(output))

print('=================================')
print('verify')
onnx.checker.check_model(args.model_file)

