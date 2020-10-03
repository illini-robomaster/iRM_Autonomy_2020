import tensorflow as tf
import numpy as np
from absl.flags import FLAGS

# maybe use tf.Tensor?
YOLO_TINY_ANCHORS = np.array(
    [[10, 14], 
    [23, 27], 
    [37, 58],
    [81, 82], 
    [135, 169],  
    [344, 319]],
    dtype = np.float32
)

YOLO_TINY_ANCHOR_MASKS = [[3,4,5], [1,2,3]]

class yoloEncoder(tf.Module):
    def __init__(self, size = 416, anchor = YOLO_TINY_ANCHORS, mask = YOLO_TINY_ANCHOR_MASKS):
        '''
        Args:
            size: size of the image
            anchor: anchor, array of shape(n, 2)
            mask: anchor id that is masked for differnent dimension,
                      For example, [[3,4,5], [1,2,3]] for yolov3-tiny
        '''
        super(yoloEncoder, self).__init__()
        self.img_size = size
        self.anchors_n2 = anchor / size
        self.anchor_masks_n3 = mask


    def transform_label_for_output(self, y_true_n6, grid_size, masks):
        '''
        Transform tensors that represent boxes into yolo_out

        Args:
            y_true: tensor of shape (boxes, (x1, y1, x2, y2, class, best_anchor))
            grid_size: int, number of 'cut' per image. For yolov3-tiny, it's 13 and 26. 
            masks: ordered anchor_ids of shape (1,n), e.g : (3,4,5)

        '''
        # mask out boxes that doesn't belong to this set of anchors
        mask = tf.math.logical_and(
            tf.math.greater_equal(y_true_n6[...,5], masks[0]), 
            tf.math.less_equal(y_true_n6[...,5], masks[-1])) 
        mask = tf.reshape(mask, (y_true_n6.shape[0],1))
        mask = tf.broadcast_to(mask, y_true_n6.shape)
        y_true_n6 = tf.boolean_mask(y_true_n6, mask)

        # y_true_out: (grid, grid, anchors, [x, y, w, h, obj, class])
        n = tf.shape(y_true_n6)[0]
        y_true_out_gg36 = tf.zeros((grid_size, grid_size, tf.shape(masks)[0], 6))
        
        # That means no box of this size
        if (n == 0):
            return y_true_out_gg36

        #box shape will be (n, 4)
        box_n4 = y_true_n6[...,0:4]
        box_yx = (y_true_n6[...,0:2] + y_true_n6[...,2:4]) / 2
        grid_yx = tf.cast(box_yx // (1/grid_size), tf.int32)

        #index should be contain info regarding which box to update
        #shape (n,3)
        #e.g.: [7,7,0] means grid 7*7 and anchor 0
        indexes_n3 = tf.stack(
            [grid_yx[...,0], 
            grid_yx[...,1], 
            tf.cast(y_true_n6[...,5] % 3, 
            dtype=tf.int32)], axis=1)
        
        #update should contain object info, size (n, 6)
        #(x, y, w, h, obj, class) at pos shape
        updates_n6 = tf.stack(
            [box_n4[...,1], 
            box_n4[...,0], 
            box_n4[...,2]-box_n4[...,0], 
            box_n4[...,3]-box_n4[...,1], 
            tf.ones((n,)), 
            y_true_n6[...,4]], axis=1)

        out = tf.tensor_scatter_nd_update(y_true_out_gg36, indexes_n3, updates_n6)
        return out


    def transform_label(self, y_train_n5):
        '''
        Read raw y_train and return yolo_out of different dimensions.

        args:
            y_train_n5: label, tensor of shape (n, (x,y,x,y,class)), should be in range(0, 1)
        '''
        y_outs = []
        grid_size = self.size // 32

        # calculate anchor index for true boxes
        anchors_n2 = tf.cast(self.anchors, tf.float32)
        anchor_area = anchors_n2[..., 0] * anchors_n2[..., 1]

        # y_true is yxyx after imgaug
        # How to calculate hw depends on tf_record format
        box_hw = y_train_n5[..., 2:4] - y_train_n5[..., 0:2]
        box_hw = tf.tile(tf.expand_dims(box_hw, -2),
                        (1, tf.shape(anchors_n2)[0], 1))
        box_area = box_hw[..., 0] * box_hw[..., 1]

        # Calculate intersection between bbox and anchors
        intersection = tf.minimum(box_hw[..., 1], anchors_n2[..., 0]) * \
            tf.minimum(box_hw[..., 0], anchors_n2[..., 1])

        # Calculate iou, which is intersection/total area
        iou = intersection / (box_area + anchor_area - intersection)
        
        # Find the best anchor
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

        # n*6
        y_train_n6 = tf.concat([y_train_n5, anchor_idx], axis=-1)

        for masks in self.anchor_masks:
            y_outs.append(self.transform_label_for_output(
                y_train_n6, grid_size, masks))
            grid_size *= 2
        
        #y_out[0] = 13*13*3*6, y_out[1] = 26*26*3*6, y_out[2] = 52*52*3*6
        # grid*grid*anchor*(x,y,w,h,confidence,class(2))
        return tuple(y_outs)

    #Placeholder for lambda function
    def transform_images_train(self, x_train):
        return x_train

    def transform_image_inference(self, x_train, size):
        x_train = tf.image.resize_with_pad(x_train, size, size)
        x_train = x_train / 255 #it's possible that we don't need this line, left it there
        return x_train