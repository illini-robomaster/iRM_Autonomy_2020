from absl import app, flags
from absl.flags import FLAGS

import os
import xml.etree.ElementTree as xmlTree

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dnn.data.converters.utils import (
    bytes_feature,
    CLASS_NAMES,
)
from dnn.utils.mem import tf_set_memory_growth

flags.DEFINE_string('input', '../DJI ROCO', 'the path for input ROCO dataset (please unzip by yourself)')
flags.DEFINE_string('output', '../DJI ROCO TFRecord', 'the path for output TFRecord file(s)')


def convert_image(image_path):
    """open an image from file path and return the image as a binary string

    Args:
        image_path:  the file path for your image
    """
    image_string = open(image_path, 'rb').read()
    return image_string


def convert_annot(annot_path):
    """open a xml file from file path and return the converted annotation

    Args:
        annot_path:  the file path for your xml annotation file
    """
    with open(annot_path, 'r') as xml_file:
        tree = xmlTree.parse(xml_file)
        root = tree.getroot()
        objt = []
        bbox = []
        for _, obj in enumerate(root.iter('object')):
            cls = obj.find('name').text
            if cls == 'ignore':
                continue
            if cls == 'armor':
                armor_color = obj.find('armor_color').text
                if armor_color == 'grey':
                    continue
                cls += '_' + armor_color
            xmlbox = obj.find('bndbox')
            objt.append(CLASS_NAMES.index(cls))
            box = [float(xmlbox.find('xmin').text),
                   float(xmlbox.find('ymin').text),
                   float(xmlbox.find('xmax').text),
                   float(xmlbox.find('ymax').text)]
            bbox.append(box)
    objt = tf.io.serialize_tensor(tf.convert_to_tensor(objt))
    bbox = tf.io.serialize_tensor(tf.convert_to_tensor(bbox))
    return objt.numpy(), bbox.numpy()


def main(_argv):
    tf_set_memory_growth()
    if not os.path.exists(FLAGS.output):
        os.mkdir(FLAGS.output)
    folder_ids = os.listdir(FLAGS.input)
    for n, folder_id in enumerate(folder_ids):
        filename = os.path.join(FLAGS.output, 'RM_training_{}.tfrecords'.format(n + 1))
        if os.path.exists(filename):
            os.remove(filename)
        writer = tf.io.TFRecordWriter(filename)
        image_ids = os.listdir(os.path.join(FLAGS.input, folder_id, 'image'))
        annot_ids = os.listdir(os.path.join(FLAGS.input, folder_id, 'image_annotation'))
        pack_ids = np.stack([image_ids, annot_ids], axis=1)
        num_image = len(pack_ids)
        for m, (image_id, annot_id) in zip(tqdm(range(num_image),
                                                desc='The {} training set: '.format(n + 1),
                                                unit='pic',
                                                ncols=150),
                                           pack_ids):
            image_target = convert_image(os.path.join(FLAGS.input, folder_id, 'image', image_id))
            object_target, bbox_target = convert_annot(os.path.join(FLAGS.input, folder_id, 'image_annotation', annot_id))
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': bytes_feature(image_target),
                'object': bytes_feature(object_target),
                'bbox': bytes_feature(bbox_target),
            }))
            writer.write(example.SerializeToString())
        writer.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
