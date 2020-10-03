import tensorflow as tf
from absl.flags import FLAGS
from dnn.data.augmentation.detection import DetectionAugmentor
from dnn.data.augmentation.image import ImageAugmentor


FEATURES = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'class_n': tf.io.FixedLenFeature([], tf.string),
    'bbox_yxyx_n4': tf.io.FixedLenFeature([], tf.string),
}


class dataLoader(tf.Module):
    def __init__(self, size = 416, feature = FEATURES):
        '''
        Args:
            !TODO modify this for a more customized data augmentation
            size: size of image
            feature_attribute: a dictionary that stores attributes of tfrecords
        '''
        super(dataLoader, self).__init__()
        self.img_size = size
        self.detection_augmentor = DetectionAugmentor(output_hw=(size, size))
        self.image_augmentor = ImageAugmentor()
        self.feature_map = feature
    
    def _parse_example(self, tfrecord):
        '''
        Takes tfrecord file, parse it, pass the data through augmentor, and return x_train, y_train
        Args:
            tfrecord: A tfrecord dataset
        '''
        example = tf.io.parse_single_example(tfrecord, self.feature_map)
        
        augmentor_input = {
            'image_hw3':    tf.io.decode_image(example['image']),
            'class_n':      tf.io.parse_tensor(example['class_n'], tf.int32),
            'bbox_yxyx_n4': tf.io.parse_tensor(example['bbox_yxyx_n4'], tf.float32),
        }
        augmentor_output = self.detection_augmentor(augmentor_input)
        
        x_train_hw3 = augmentor_output['image_hw3']
        x_train_hw3 = self.image_augmentor(x_train_hw3)
        
        y_train_n5 = tf.concat(
            [augmentor_output['bbox_yxyx_n4']/self.img_size, 
            tf.expand_dims(augmentor_output['label_n'], -1)], 
            axis = -1)

        return x_train_hw3, y_train_n5

    def __call__(self, file):
        '''
        takes tfrecord file and output dataset
        Args:
            file: A file (a list of files?) of tfrecord
        '''
        return tf.data.TFRecordDataset(file).map(lambda x : self._parse_example(x))

