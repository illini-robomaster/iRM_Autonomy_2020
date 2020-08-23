"""
This is a converter for the Purdue Dataset enhanced by iRM
Input raw data can be downloaded from the following URL:

https://drive.google.com/file/d/1MarfJDgRdSPM_OvfqyZSYlDXhG7cb4Wq/view?usp=sharing
"""

import json
import io
import os
import random
import tensorflow as tf

from absl import app, flags
from absl.flags import FLAGS
from PIL import Image
from tqdm import trange, tqdm

from dnn.data.converters.utils import (
    bytes_feature,
    CLASS_NAMES,
)
from dnn.utils.mem import tf_set_memory_growth

flags.DEFINE_string('input', None, 'input directory to purdue dataset iRM version')
flags.DEFINE_string('output', None, 'output directory to store TFRecord file(s)')
flags.DEFINE_integer('shard_size', 1000, 'number of samples per TFRecord shard')
flags.DEFINE_float('split', .9, 'train / validation split, e.g. .9 for .9 train and .1 validation',
                    lower_bound=0., upper_bound=1.)


def generate_data_split(samples, class_names, mode='train'):
    num_shards = len(samples) // FLAGS.shard_size
    output_dir = os.path.join(FLAGS.output, mode)
    os.makedirs(output_dir, exist_ok=True)
    for shard_id in trange(num_shards, desc=f'{mode} shards'):
        shard_path = os.path.join(output_dir, f'shard-{shard_id}.tfrecord')
        with tf.io.TFRecordWriter(shard_path) as writer:
            for sample in tqdm(samples[shard_id*FLAGS.shard_size:(shard_id+1)*FLAGS.shard_size],
                               desc=f'shard {shard_id}', position=1, leave=False):
                # get image
                img_path = os.path.join(FLAGS.input, 'data', f'{sample}.png')
                img = Image.open(img_path)
                with open(img_path, 'rb') as f:
                    img_bin = f.read()
                # get annotation
                anno_path = os.path.join(FLAGS.input, 'annotation', f'{sample}.txt')
                with open(anno_path, 'r') as f:
                    annotations = f.read().split('\n')
                # parse annotation
                class_n = []
                bbox_yxyx_n4 = []
                for annotation in annotations:
                    if not annotation:
                        continue
                    class_name = class_names[int(annotation.split(' ')[0])]
                    if class_name not in CLASS_NAMES: # class not of interests
                        continue
                    class_id = CLASS_NAMES.index(class_name)
                    class_n.append(class_id)
                    x_center, y_center, w, h = [float(val) for val in annotation.split(' ')[1:]]
                    x = (x_center - w / 2) * img.width
                    y = (y_center - h / 2) * img.height
                    w *= img.width
                    h *= img.height
                    bbox_yxyx_n4.append([y, x, y + h, x + w])
                # to tf example
                class_n = tf.convert_to_tensor(class_n, dtype=tf.int32)
                bbox_yxyx_n4 = tf.convert_to_tensor(bbox_yxyx_n4, dtype=tf.float32)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': bytes_feature(img_bin),
                    'class_n': bytes_feature(tf.io.serialize_tensor(class_n).numpy()),
                    'bbox_yxyx_n4': bytes_feature(tf.io.serialize_tensor(bbox_yxyx_n4).numpy())
                }))
                writer.write(example.SerializeToString())

def main(_argv):
    tf_set_memory_growth()
    # ensure output path exists
    os.makedirs(FLAGS.output, exist_ok=True)
    # read in sample annotations
    samples = [s.split('.')[0] for s in os.listdir(os.path.join(FLAGS.input, 'annotation'))]
    with open(os.path.join(FLAGS.input, 'classes.names'), 'r') as f:
        class_names = f.read().split('\n')
    # train / validation split
    random.shuffle(samples)
    split_idx = int(len(samples) * FLAGS.split)
    generate_data_split(samples[:split_idx], class_names, 'train')
    generate_data_split(samples[split_idx:], class_names, 'validation')

if __name__ == '__main__':
    app.run(main)
