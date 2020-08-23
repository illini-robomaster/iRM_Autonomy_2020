import os
import xml.etree.ElementTree as xmlTree

import numpy as np
import tensorflow as tf

from absl import app, flags
from absl.flags import FLAGS
from PIL import Image
from tqdm import tqdm

from dnn.data.converters.utils import bytes_feature, CLASS_NAMES
from dnn.utils.mem import tf_set_memory_growth

flags.DEFINE_string('input', None, 'the path for input ROCO dataset (please unzip by yourself)')
flags.DEFINE_string('output', None, 'the path for output TFRecord file(s)')


def convert_image(image_path):
    """open an image from file path and return the image as a binary string

    Args:
        image_path:  the file path for your image
    """
    with open(image_path, 'rb') as image:
        image_string = image.read()
    return image_string


def convert_annot(annot_path, image_path):
    """open a xml file from file path and return the converted annotation

    Args:
        annot_path:  the file path for your xml annotation file
        image_path:  path to the associated image
    """
    img = Image.open(image_path)
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
        bndbox = obj.find('bndbox')
        objt.append(CLASS_NAMES.index(cls))
        box = [float(bndbox.find('xmin').text) * img.width,
               float(bndbox.find('ymin').text) * img.height,
               float(bndbox.find('xmax').text) * img.width,
               float(bndbox.find('ymax').text) * img.height]
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
        output_path = os.path.join(FLAGS.output, 'ROCO_{}.tfrecords'.format(n + 1))
        if os.path.exists(output_path):
            os.remove(output_path)
        with tf.io.TFRecordWriter(output_path) as writer:
            image_ids = os.listdir(os.path.join(FLAGS.input, folder_id, 'image'))
            annot_ids = os.listdir(os.path.join(FLAGS.input, folder_id, 'image_annotation'))
            num_image = len(pack_ids)
            for image_id, annot_id in tqdm(zip(image_ids, annot_ids)):
                image_path = os.path.join(FLAGS.input, folder_id, 'image', image_id)
                annot_path = os.path.join(FLAGS.input, folder_id, 'image_annotation', annot_id)
                image_target = convert_image(image_path)
                object_target, bbox_target = convert_annot(annot_path, image_path)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': bytes_feature(image_target),
                    'class_n': bytes_feature(object_target),
                    'bbox_yxyx_n4': bytes_feature(bbox_target),
                }))
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    app.run(main)
