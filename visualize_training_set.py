import os
import tensorflow as tf
import sys
import cv2
import numpy as np

class_names = ['car', 'watcher', 'base', 'armor_red', 'armor_blue']

IMAGE_FEATURE_MAP = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}


@tf.function
def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, IMAGE_FEATURE_MAP)


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

folder_ids = os.listdir('./data/DJI ROCO TFRecord/')
for n, training_id in enumerate(folder_ids):
    training_data = tf.data.TFRecordDataset('./data/DJI ROCO TFRecord/' + training_id)
    data_set = training_data.map(_parse_image_function)
    for m, image_features in enumerate(data_set):
        tf.print('The {} picture:'.format(m + 1), output_stream=sys.stdout)
        image_raw = tf.io.decode_jpeg(image_features['image/encoded'], channels=3)
        objects = tf.sparse.to_dense(image_features['image/object/class/text'])
        x1 = tf.sparse.to_dense(image_features['image/object/bbox/xmin'])
        y1 = tf.sparse.to_dense(image_features['image/object/bbox/ymin'])
        x2 = tf.sparse.to_dense(image_features['image/object/bbox/xmax'])
        y2 = tf.sparse.to_dense(image_features['image/object/bbox/ymax'])
        boxes = []
        scores = []
        classes = []
        for p, label in enumerate(objects):
            boxes.append((x1[p], y1[p], x2[p], y2[p]))
            scores.append(1)
            classes.append(class_names.index(objects[p]))
        nums = [len(boxes)]
        boxes = [boxes]
        scores = [scores]
        classes = [classes]
        img = cv2.cvtColor(image_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imshow("RM Training Set", img)
        cv2.waitKey(0)
