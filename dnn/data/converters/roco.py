"""
This is a converter for the DJI ROCO dataset
Raw dataset can be downloaded from either one of the sources below

1. DJI (prone to error / change):
    https://terra-1-g.djicdn.com/b2a076471c6c4b72b574a977334d3e05/resources/DJI%20ROCO.zip

2. iRM OneDrive:
    https://uillinoisedu-my.sharepoint.com/:u:/g/personal/yixiaos3_illinois_edu/Ef5LIWMYRpNLgMuXGHOrAZoBVKdsymnJ8xWinbJO_pX9OQ?e=POF0Ts
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.').split('/dnn')[0])
print('Trim path to',sys.path[0])

from dnn.utils.mem import tf_set_memory_growth
from dnn.data.converters.utils import bytes_feature, CLASS_NAMES
import os
import xml.etree.ElementTree as xmlTree

import numpy as np
import tensorflow as tf

from absl import app, flags
from absl.flags import FLAGS
from PIL import Image
from tqdm import tqdm


flags.DEFINE_string(
    'input', None, 'the path for input ROCO dataset (please unzip by yourself)')
flags.DEFINE_string('output', None, 'the path for output TFRecord file(s)')


def convert_annot(annot_path):
    """open a xml file from file path and return the converted annotation

    Args:
        annot_path:  the file path for your xml annotation file
    """
    # parse annotation
    with open(annot_path, 'r') as xml_file:
        tree = xmlTree.parse(xml_file)
    root = tree.getroot()
    objt = []
    bbox = []
    difficulties = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls == 'ignore':
            continue
        elif cls == 'armor':
            armor_color = obj.find('armor_color').text
            if armor_color == 'grey':
                continue
            cls += '_' + armor_color
        bndbox = obj.find('bndbox')
        objt.append(CLASS_NAMES.index(cls))
        box = [float(bndbox.findtext('ymin')),
               float(bndbox.findtext('xmin')),
               float(bndbox.findtext('ymax')),
               float(bndbox.findtext('xmax'))]
        bbox.append(box)
        diff = obj.findtext('difficulty')
        diff = int(diff) if diff is not None else -1
        difficulties.append(diff)
    objt = tf.io.serialize_tensor(tf.convert_to_tensor(objt, dtype=tf.int32))
    bbox = tf.io.serialize_tensor(tf.convert_to_tensor(bbox, dtype=tf.float32))
    difficulties = tf.io.serialize_tensor(tf.convert_to_tensor(difficulties, dtype=tf.int8))
    # get encoded image binaries
    image_path = annot_path.replace(
        'image_annotation', 'image').replace('.xml', '.jpg')
    with open(image_path, 'rb') as f:
        image_target = f.read()

    return image_target, objt.numpy(), bbox.numpy(), difficulties.numpy()


def main(_argv):
    tf_set_memory_growth()
    os.makedirs(FLAGS.output, exist_ok=True)
    folder_ids = os.listdir(FLAGS.input)
    for n, folder_id in enumerate(folder_ids):
        if not os.path.isdir(os.path.join(FLAGS.input, folder_id)):  # skip irrelavent files
            continue
        output_path = os.path.join(
            FLAGS.output, 'ROCO_{}.tfrecords'.format(n + 1))
        if os.path.exists(output_path):
            os.remove(output_path)
        annot_dir = os.path.join(FLAGS.input, folder_id, 'image_annotation')
        with tf.io.TFRecordWriter(output_path) as writer:
            for annot_file in os.listdir(annot_dir):
                annot_path = os.path.join(annot_dir, annot_file)
                image_target, object_target, bbox_target, difficulties = convert_annot(
                    annot_path)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': bytes_feature(image_target),
                    'class_n': bytes_feature(object_target),
                    'bbox_yxyx_n4': bytes_feature(bbox_target),
                    '''
                    difficulty:
                        -1: DNE
                        0：cover<=20%
                        1，20%<cover<=50%
                        2，cover>50%
                    '''
                    'difficulties': bytes_feature(difficulties),
                }))
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    app.run(main)
