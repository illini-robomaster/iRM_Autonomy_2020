import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from absl import flags, app
from absl.flags import FLAGS
from tqdm import tqdm

from dnn.utils.dataLoader import dataLoader
from dnn.model.yolov3_tiny import YoloV3Tiny
from dnn.utils.mem import tf_set_memory_growth
tf_set_memory_growth()

flags.DEFINE_string('weights', './checkpoints/yolov3_train_49.tf',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('val_data', 'F:/RM/iRM_Autonomy_2020/dnn/data/roco/ROCO_5.tfrecords',
                    'path to tfrecord dataset, should be in same format defined in converter')
flags.DEFINE_integer('min_iou', 50, 'minimum iou to consider as positive')
flags.DEFINE_bool('coco_map', True, 'calculate coco mAP')
flags.DEFINE_integer('num_classes', 5, 'number of classes in the model')
CLASS_NAME = ['car', 'watcher', 'base', 'armor_red', 'armor_blue']

def main(_argv):
    yolo = YoloV3Tiny(size = FLAGS.size, classes = FLAGS.num_classes, training = False)
    yolo.load_weights(FLAGS.weights)

    loader = dataLoader(size = FLAGS.size, train = False)
    data = loader(FLAGS.val_data)
    
    if FLAGS.coco_map:
        thresholds = np.arange(0.5, 1, 0.05)
        result = np.zeros((len(thresholds), FLAGS.num_classes))
        for (idx, t) in enumerate(thresholds):
            temp_n3 = calculate_map(yolo, data, t, True)
            result[idx] = temp_n3[:,0] / (temp_n3[:,0] + temp_n3[:,1])
        for (idx, c) in enumerate(CLASS_NAME):
            print("AP for class " + c + " " + str(np.mean(result[:,idx])))
        print("mAP is " + str(np.mean(result)))
    else:
        calculate_map(yolo, data, FLAGS.min_iou / 100)

def calculate_map(model, data, iou_threshold, coco_map = False):
    # num_cls * (tp, fp, fn)
    pred_record = np.zeros((FLAGS.num_classes,3))
    
    for img, label in tqdm(iter(data)):
        boxes, scores, classes, nums = model.predict(tf.expand_dims(img, 0))
        pred_n5 = np.concatenate([boxes, np.expand_dims(classes, 2)], 2)
        # x,y,x,y,cls,confidence
        pred_n5 = pred_n5[:,:nums[0],:]
        pred_n5 = np.squeeze(pred_n5, 0)
        # convert label back back to xyxyc
        label_yxyxc = np.array(label)
        used = np.zeros((label_yxyxc.shape[0]))
        label_n6 = np.stack([label_yxyxc[:,1], label_yxyxc[:,0], label_yxyxc[:,3], label_yxyxc[:,2], label_yxyxc[:,4], used])
        label_n6 = label_n6.T

        for pred_5 in pred_n5:
            max_iou = -1
            match_idx = -1
            for (idx, obj_6) in enumerate(label_n6):
                if pred_5[4] == obj_6[4] and obj_6[5] == 0:
                    x1 = max(pred_5[0], obj_6[0])
                    y1 = max(pred_5[1], obj_6[1])
                    x2 = min(pred_5[2], obj_6[2])
                    y2 = min(pred_5[3], obj_6[3])
                    intersect = (x2 - x1) * (y2 - y1)
                    iou = intersect / ((obj_6[3] - obj_6[1]) * (obj_6[2] - obj_6[0]) + (pred_5[3] - pred_5[1]) * (pred_5[2] - pred_5[0]) - intersect)
                    if iou > max_iou:
                        max_iou = iou
                        match_idx = idx
            if max_iou > iou_threshold:
                # true positive
                label_n6[idx, 5] = 1
                pred_record[int(label_n6[idx, 4]), 0] += 1
            else:
                # false postitive
                pred_record[int(pred_5[4]), 1] += 1
        
        # false negative
        for obj_6 in label_n6:
            if obj_6[5] == 0:
                pred_record[int(obj_6[4]), 2] += 1
        
    # Print result
    if coco_map:
        return pred_record
    else:
        for (idx, record) in enumerate(pred_record):
            print("Precision for class " + CLASS_NAME[idx] + " " + str(record[0] / (record[0] + record[1])))
            print("Recall for class " + CLASS_NAME[idx] + " " + str(record[0] / (record[0] + record[2])))
        

if __name__ == '__main__':
    app.run(main)