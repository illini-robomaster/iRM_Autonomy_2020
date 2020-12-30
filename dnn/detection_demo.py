import time
import cv2
import tensorflow as tf
from absl import flags, app
from absl.flags import FLAGS

from dnn.utils.mem import tf_set_memory_growth
tf_set_memory_growth()
from dnn.model.yolov3_tiny import YoloV3Tiny
from dnn.utils.inference_utils import draw_inference

CLASS_NAME = ['car', 'watcher', 'base', 'armor_red', 'armor_blue']

flags.DEFINE_string('weights', './checkpoints/yolov3_train_100.tf',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', r'F:\RM\videos.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', './out.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 5, 'number of classes in the model')

def main(_argv):
    yolo = YoloV3Tiny(size=FLAGS.size, classes=FLAGS.num_classes, training=False)
    yolo.load_weights(FLAGS.weights)
    print('Weight loaded')

    times = []
    
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = vid.read()

        if img is None:
            print("End of video")
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.image.resize(img_in, (FLAGS.size, FLAGS.size))
        img_in = tf.expand_dims(img_in, 0)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_inference(img, (boxes, scores, classes, nums), CLASS_NAME)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(main)