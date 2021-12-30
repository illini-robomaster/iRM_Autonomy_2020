import cv2
import onnxruntime as rt
import numpy as np
import argparse
import time

from dnn.utils.inference_utils import draw_inference

CLASS_NAME = ['car', 'watcher', 'base', 'armor_red', 'armor_blue']

parser = argparse.ArgumentParser(description='Inspect an onnx model.')
parser.add_argument('--model_file', type=str, default='./save_onnx/model.onnx', help='onnx model file to use')
parser.add_argument('--video_file', type=str, default='./sanity.mp4', help='video model file to use')
parser.add_argument('--display', type=int, default=1, help='display video on screen')
parser.add_argument('--size', type=int, default=416, help='resize images to')
args = parser.parse_args()

print(f'Load model from: {args.model_file}')
output_names = ['yolo_nms','yolo_nms_1','yolo_nms_2','yolo_nms_3']
providers = ['CUDAExecutionProvider','CPUExecutionProvider']
m = rt.InferenceSession(args.model_file, providers=providers)

# dummy input
# x = np.zeros([1,416,416,3]).astype(np.float32)
# onnx_pred = m.run(output_names, {"input": x})
# print('ONNX Predicted:', onnx_pred)

def inference(x):
    onnx_pred = m.run(output_names, {"input": x})
    return onnx_pred

def main():
    times = []

    try:
        vid = cv2.VideoCapture(int(args.video_file))
    except:
        vid = cv2.VideoCapture(args.video_file)

    out = None

    while True:
        _, img = vid.read()

        if img is None:
            print("End of video")
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_in = cv2.resize(img_in, (args.size, args.size), interpolation = cv2.INTER_AREA)
        img_in = np.expand_dims(img_in, 0)

        t1 = time.time()
        boxes, scores, classes, nums = inference(img_in)
        t2 = time.time()
        print(f'fps:{1./(t2-t1)}')
        times.append(t2-t1)
        times = times[-20:]

        if args.display:
            img = draw_inference(img, (boxes, scores, classes, nums), CLASS_NAME)
            img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv2.imshow('output', img)

            if cv2.waitKey(1) == ord('q'):
                break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

