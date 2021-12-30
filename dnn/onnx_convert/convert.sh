cd save_onnx
python3 -m tf2onnx.convert --saved-model save_model_format/ --output model.onnx --opset 12
python3 -c "import onnx;onnx.checker.check_model('model.onnx')"
