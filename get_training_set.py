import xml.etree.ElementTree as xmlTree
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

class_names = ['car', 'watcher', 'base', 'armor_red', 'armor_blue']


@tf.function
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert(size, box):
    x1 = box[0] / size[0]
    y1 = box[1] / size[1]
    x2 = box[2] / size[0]
    y2 = box[3] / size[1]
    return [x1, y1, x2, y2]


def convert_image(image_path):
    image_string = open(image_path, 'rb').read()
    return image_string


def convert_annot(annot_path):
    xml_file = open(annot_path)
    tree = xmlTree.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    objt = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    n = 0
    for i, obj in enumerate(root.iter('object')):
        n += 1
        cls = obj.find('name').text
        if cls == 'ignore':
            continue
        if cls == 'armor':
            armor_color = obj.find('armor_color').text
            if armor_color == 'grey':
                continue
            cls += '_' + armor_color
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymax').text))
        objt.append(cls.encode('utf8'))
        bbox_data = convert((w, h), b)
        x1.append(bbox_data[0])
        y1.append(bbox_data[1])
        x2.append(bbox_data[2])
        y2.append(bbox_data[3])
    xml_file.close()
    return objt, x1, y1, x2, y2


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

folder_ids = os.listdir('./data/DJI ROCO/')
for n, folder_id in enumerate(folder_ids):
    filename = os.path.join('./data/DJI ROCO TFRecord/' + 'RM_training_{}'.format(n + 1) + '.tfrecords')
    writer = tf.io.TFRecordWriter(filename)
    image_ids = os.listdir('./data/DJI ROCO/' + folder_id + '/image')
    annot_ids = os.listdir('./data/DJI ROCO/' + folder_id + '/image_annotation')
    pack_ids = np.stack([image_ids, annot_ids], axis=1)
    num_image = len(pack_ids)
    for m, (image_id, annot_id) in zip(tqdm(range(num_image),
                                            desc='The {} training set: '.format(n + 1),
                                            unit='pic',
                                            ncols=100),
                                       pack_ids):
        image_target = convert_image('./data/DJI ROCO/' + folder_id + '/image/' + image_id)
        object_target, x1_target, y1_target, x2_target, y2_target = convert_annot(
            './data/DJI ROCO/' + folder_id + '/image_annotation/' + annot_id)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(image_target),
            'image/object/bbox/xmin': float_list_feature(x1_target),
            'image/object/bbox/ymin': float_list_feature(y1_target),
            'image/object/bbox/xmax': float_list_feature(x2_target),
            'image/object/bbox/ymax': float_list_feature(y2_target),
            'image/object/class/text': bytes_list_feature(object_target),
        }))
        writer.write(example.SerializeToString())
    writer.close()
