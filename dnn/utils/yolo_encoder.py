import tensorflow as tf
from absl.flags import FLAGS
from dnn.data.augmentation.augmentor import DetectionAugmentor

'''
This file is directly taken from https://github.com/yihjian/yolov3_tf2/blob/master/yolov3_tf2/dataset.py
!TODO:
1. Adapt to new data pipeline
2. Adapt to new tfrecord format
3. Vectorize

Old data flow:
1. read df record
2. aug image 
3. create a tf.dataset
4. use lambda functions on tf.dataset to call transform_targets and transform_image (not sure why we need transform_img, got no reply in issue)
5. transform_targets call transform_target_for_output
'''

@tf.function
def transform_targets_for_output_vectorize(y_true, grid_size, masks):
    '''
    This doesn't work yet
    '''
    # y_true: (boxes, (x1, y1, x2, y2, class, best_anchor))
    # y_true_out: (grid, grid, anchors, [x, y, w, h, obj, class])
    n = tf.shape(y_true)[0]
    y_true_out = tf.zeros((grid_size, grid_size, tf.shape(masks)[0], 6))
    
    # Because anchors are devided into two/three groups of different grid size,
    # we need to convert anchor id from 0-8/0-5 to 0-2
    current_size_anchor_id = tf.cast(masks, tf.int32)
    #expand it into [3, n]
    current_size_anchor_id = tf.tile(
        tf.expand_dims(current_size_anchor_id,1), 
        [1,n]
    )
    anchor_eq = tf.equal(current_size_anchor_id, tf.cast(y_true[...,5], tf.int32))
    anchor_id = tf.cast(tf.where(anchor_eq), tf.int32)

    if(tf.equal(tf.shape(anchor_id)[0], 0)):
        return y_true
    #box shape will be (n, 4)
    box = y_true[...,0:4]
    box_yx = (y_true[...,0:2] + y_true[...,2:4]) / 2
    grid_yx = tf.cast(box_yx // (1/grid_size), tf.int32)

    #index should be contain info regarding which box to update
    #shape (n,3)
    #e.g.: [7,7,0] means grid 7*7 and anchor 0
    indexes = tf.stack(
        [grid_yx[...,0], 
         grid_yx[...,1], 
         tf.cast(anchor_id[...,0], 
         dtype=tf.int32)], axis=1)
    
    #update should contain object info, size (n, 6)
    #(x, y, w, h, obj, class) at pos shape
    updates = tf.stack(
        [box[...,1], 
         box[...,0], 
         box[...,2]-box[...,0], 
         box[...,3]-box[...,1], 
         tf.ones((n,)), 
         y_true[...,4]], axis=1)

    out = tf.tensor_scatter_nd_update(y_true_out, indexes, updates)
    print(tf.shape(out))
    return out

def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    '''
    This works, but pretty slow
    '''
    # y_true: (boxes, (x1, y1, x2, y2, class, best_anchor))
    # n*6
    N = tf.shape(y_true)[0]

    # y_true_out: (grid, grid, anchors, [x, y, w, h, obj, class])
    # 13*13*3*6
    y_true_out = tf.zeros(
        (grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for j in tf.range(N):
        # anchor 0-5
        # mask out if not belong to this size
        anchor_eq = tf.equal(
            anchor_idxs, tf.cast(y_true[j][5], tf.int32))

        # If belong to this dimension
        if tf.reduce_any(anchor_eq):
            box = y_true[j][0:4]
            box_yx = (y_true[j][0:2] + y_true[j][2:4]) / 2

            anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
            grid_yx = tf.cast(box_yx // (1/grid_size), tf.int32)

            # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
            indexes = indexes.write(
                    idx, [grid_yx[0], grid_yx[1], anchor_idx[0][0]]) #  1, 7, 7, 0
            updates = updates.write(
                    idx, [box[1], box[0], box[3], box[2], 1, y_true[j][4]]) #1, xmin, ymin, xmax, ymax, 1, class
            idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    # update y_true[7][7][0] to (1, xmin, ymin, xmax, ymax, 1, class)
    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())    


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    # anchors shape 3*2
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]

    # y_true is yxyx after imgaug
    box_hw = y_train[..., 2:4] - y_train[..., 0:2]
    box_hw = tf.tile(tf.expand_dims(box_hw, -2),
                     (1, tf.shape(anchors)[0], 1))
    box_area = box_hw[..., 0] * box_hw[..., 1]

    # Calculate intersection between bbox and anchors
    intersection = tf.minimum(box_hw[..., 1], anchors[..., 0]) * \
        tf.minimum(box_hw[..., 0], anchors[..., 1])

    # Calculate iou, which is intersection/total area
    iou = intersection / (box_area + anchor_area - intersection)
    
    # Find the best anchor
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    # n*6
    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    #y_out[0] = 13*13*3*6, y_out[1] = 26*26*3*6, y_out[2] = 52
    # grid*grid*anchor*(x,y,w,h,confidence,class(2))
    # 13*13*3*7 [1]
    return tuple(y_outs)

#Placeholder for lambda function
'''
Not sure why is it necessary to resize again
'''
def transform_images(x_train, size):
    x_train = tf.image.resize_with_pad(x_train, size, size)
    x_train = x_train / 255
    return x_train


'''
Following function need to be changed to adapt to new tfrecord format
'''
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}

def parse_tfrecord(tfrecord, class_table, size, augmentor):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    
    img = tf.image.decode_png(x['image/encoded'], channels=3)
    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    boxes = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax'])], axis=1)
    input_data = {'image_hw3': img, 'bbox_yxyx_n4': boxes, 'label_n': labels}
    
    output_data = augmentor(input_data)
    
    x_train = output_data['image_hw3']
    # Format ymin, xmin, ymax, xmax
    # y_out will be num_box*5
    # shape = n*5 (ymin, xim, ymax, xmax, class)
    y_train = tf.concat([output_data['bbox_yxyx_n4']/size, tf.expand_dims(output_data['label_n'], -1)], axis = -1)
    
    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file, size=416):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    augmentor = DetectionAugmentor(output_hw=(416, 416))
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size, augmentor))


# def load_fake_dataset():
#     x_train = tf.image.decode_jpeg(
#         open('./data/girl.png', 'rb').read(), channels=3)
#     x_train = tf.expand_dims(x_train, axis=0)

#     labels = [
#         [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
#         [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
#         [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
#     ] + [[0, 0, 0, 0, 0]] * 5
#     y_train = tf.convert_to_tensor(labels, tf.float32)
#     y_train = tf.expand_dims(y_train, axis=0)

#     return tf.data.Dataset.from_tensor_slices((x_train, y_train))
