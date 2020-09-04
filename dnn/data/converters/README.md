# Detection Data TFRecord Converter Format

Different data sources will be converted into a unified format as
[TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord). This
document will describe what the format is, and what could be a minimal example
for loading data from the converted dataset.

## Format Specification

Each encoded example will contain the following keys

```python
'image' -> tf.string        # jpeg / png encoded image data
'class_n' -> tf.string      # object classes
'bbox_yxyx_n4' -> tf.string # bounding boxes in (y1, x1, y2, x2) format
```

**Note**: the nameing convention for the post fix `_xyz` simply means a
serialized tensor of shape (x, y, z).

## An Example Parser

The following code snippet will be a minimal example to construct a TF Dataset.

```python
import tensorflow as tf

def _parse_example(example_proto):
    example = tf.io.parse_single_example(example_proto, {
        'image': tf.io.FixedLenFeature([], tf.string),
        'class_n': tf.io.FixedLenFeature([], tf.string),
        'bbox_yxyx_n4': tf.io.FixedLenFeature([], tf.string),
    })

    return {
        'image_hw3':    tf.io.decode_image(example['image']),
        'class_n':      tf.io.parse_tensor(example['class_n'], tf.int32),
        'bbox_yxyx_n4': tf.io.parse_tensor(example['bbox_yxyx_n4'], tf.float32),
    }

dataset = tf.data.TFRecordDataset(<list of xxx.tfrecord>).map(_parse_example)
# do stuff with the dataset
```
