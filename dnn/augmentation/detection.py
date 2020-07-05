import math
import tensorflow as tf
import tensorflow_addons as tfa

def get_affine_coeffs(output_hw, center_yx, scale_yx):
    """generate affine coefficients for tfa.image.transform

    Args:
        output_hw:  output image size in (height, width)
        center_yx:  center coordinates of the crop in (y, x)
        scale_yx:   scale of the crop in each dimension (scale_y, scale_x)
                    scale is defined as output_hw / input_hw, which means
                    the larger the scale, the more "zoomed in" the crop
    """
    # top-left to center offset in output space
    offset_x = output_hw[1] / 2
    offset_y = output_hw[0] / 2
    # calculate affine parameters
    a0 = 1 / scale_yx[1]
    a1 = 0.
    a2 = center_yx[1] - a0 * offset_x
    b0 = 0.
    b1 = 1 / scale_yx[0]
    b2 = center_yx[0] - b1 * offset_y
    return (a0, a1, a2, b0, b1, b2, 0., 0.)


class DetectionAugmentor(tf.Module):
    """
    For input format, you could

    1. inherit this class and override the following method to extract certain attributes
        get_image_hw3
        get_bbox_yxyx_n4
        get_label_n

    2. ensure the input data is in a dictionary with following format
        {
            'image_hw3': image tensor with shape (h, w, 3)
            'bbox_yxyx_n4': n bounding boxes in YXYX format
            'label_n': n labels associated with the bboxes
        }

    For output format, you could

    1. inherit this class and override the following method to create customized output
        format_output

    2. accept the default format as specified above
    """

    def __init__(self, 
            output_hw=(256, 256), 
            min_aspect=3/4, 
            max_zoom=2, 
            min_bbox_area=.5, 
            focus=0.5, 
            focus_jitter=50, 
            focus_scale=(0.2, 1.)):
        """ 
        Args:
            output_hw:  output image size (height, width)
            min_aspect:     minimum aspect ratio range for random streching / squeezing
                            max_aspect = 1 / min_aspect
            max_zoom:       maximum zoom scale (to prevent from extreme pixelation due to
                                                zooming in too much)
            min_bbox_area:  minimum visable portion (after crop) of a bounding box to be retained 
            focus:          probability to focus the crop box around a randomly selected instance
                            as oppose to generating a random crop from the entire image
            focus_jitter:   amount of random translation (# pixels in output space)
            focus_scale:    a tuple (min_focus_scale, max_focus_scale) indicating how big the 
                            crop box is (w.r.t. a crop box tightly bounding the focus instance).
                            similar to all definition of scales, the higher the value, the 
                            more "zoomed in" the crop box
                            e.g. 
                                1. a value of 1 represents the size of a crop box that tightly
                                   bounds the object
                                2. a value of 0.5 represents twice the size of a crop box that
                                   tightly bounds the object

        """
        super(DetectionAugmentor, self).__init__()
        self.output_hw = tf.convert_to_tensor(output_hw, dtype=tf.float32)
        self.log_min_aspect = math.log(min_aspect)
        self.max_zoom = max_zoom
        self.min_bbox_area = min_bbox_area
        self.focus = focus
        self.focus_jitter = focus_jitter
        self.focus_log_scale = (math.log(focus_scale[0]), math.log(focus_scale[1]))

    def get_image_hw3(self, data):
        """ override this for customized input data format """
        return data['image_hw3']

    def get_bbox_yxyx_n4(self, data):
        """ override this for customized input data format """
        return tf.cast(data['bbox_yxyx_n4'], dtype=tf.float32)

    def get_label_n(self, data):
        """ override this for customized input data format """
        return data['label_n']

    def format_output(self, image_hw3, bbox_yxyx_n4, label_n):
        """ override this for customized output data format """
        return {
            'image_hw3': image_hw3,
            'bbox_yxyx_n4': bbox_yxyx_n4,
            'label_n': label_n,
        }

    def get_clipped_scale_yx(self, input_hw, zoom):
        zoom = tf.minimum(zoom, self.max_zoom)
        scale_yx_min = self.output_hw / input_hw
        # aspect ratio squeeze / stretch
        aspect_ratio = tf.exp(tf.random.uniform([], self.log_min_aspect, -self.log_min_aspect))
        area_scale = zoom ** 2
        scale_y = tf.sqrt(area_scale * aspect_ratio)
        scale_x = area_scale / scale_y
        # clip dimensions larger than original image
        scale_multiplier = tf.maximum(scale_yx_min[0] / scale_y, scale_yx_min[1] / scale_x)
        if scale_multiplier > 1.:
            scale_y *= scale_multiplier
            scale_x *= scale_multiplier
        return tf.convert_to_tensor([scale_y, scale_x])

    def get_full_img_affine(self, input_hw):
        scale_yx_min = self.output_hw / input_hw
        min_zoom = tf.reduce_max(scale_yx_min)
        # avoid magnification (i.e. zooming in) when performing full image crop
        if min_zoom < 1.:
            zoom = tf.exp(tf.random.uniform([], tf.math.log(min_zoom), 0.))
        else:
            zoom = min_zoom
        scale_yx = self.get_clipped_scale_yx(input_hw, zoom)
        input_crop_hw = self.output_hw / scale_yx
        # randomly sample valid crop center locations
        center_yx = tf.random.uniform([], [0, 0], input_hw - input_crop_hw) + input_crop_hw / 2
        return center_yx, scale_yx

    def get_focus_affine(self, input_hw, bbox_yxyx_4):
        bbox_hw_2 = bbox_yxyx_4[2:] - bbox_yxyx_4[:2]
        scale_focus_yx = self.output_hw / bbox_hw_2
        ref_zoom = tf.reduce_min(scale_focus_yx)
        zoom = ref_zoom * tf.exp(tf.random.uniform(
            [], self.focus_log_scale[0], self.focus_log_scale[1]))
        scale_yx = self.get_clipped_scale_yx(input_hw, zoom)
        # find a valid crop center
        input_crop_hw = self.output_hw / scale_yx
        bbox_center_yx = bbox_yxyx_4[:2] + bbox_hw_2 / 2
        center_yx_min = tf.clip_by_value(bbox_center_yx - self.focus_jitter / scale_yx,
            input_crop_hw / 2, input_hw - input_crop_hw / 2)
        center_yx_max = tf.clip_by_value(bbox_center_yx + self.focus_jitter / scale_yx,
            input_crop_hw / 2, input_hw - input_crop_hw / 2)
        center_yx = tf.random.uniform([], center_yx_min, center_yx_max)
        return center_yx, scale_yx

    def get_random_affine(self, input_hw, bbox_yxyx_n4):
        if bbox_yxyx_n4.shape[0] and tf.random.uniform([]) < self.focus:
            # sample a focus instance (the larger the size, the more likely to be sampled)
            bbox_hw_n2 = bbox_yxyx_n4[:, 2:] - bbox_yxyx_n4[:, :2]
            bbox_size_n = tf.reduce_prod(bbox_hw_n2, axis=-1)
            idx = tf.random.categorical(tf.math.log(bbox_size_n)[tf.newaxis], 1)[0, 0]
            return self.get_focus_affine(input_hw, bbox_yxyx_n4[idx])
        else:
            return self.get_full_img_affine(input_hw)

    def get_center_affine(self, input_hw):
        center_yx = tf.cast(input_hw, tf.float32) / 2
        scale_yx = self.output_hw / input_hw
        scale = tf.reduce_max(scale_yx)
        return center_yx, (scale, scale)

    def affine_transform_bbox(self, affine_coeffs, bbox_yxyx_n4): 
        # calculate original bbox sizes
        bbox_hw_old_n2 = bbox_yxyx_n4[:, 2:] - bbox_yxyx_n4[:, :2]
        bbox_size_old_n = tf.reduce_prod(bbox_hw_old_n2, axis=-1)
        # put bbox into homogenous coordinates
        bbox_yxyx_n22 = tf.reshape(bbox_yxyx_n4, (-1, 2, 2))
        ones_yxyx_n21 = tf.ones_like(bbox_yxyx_n22[..., 0])[..., tf.newaxis]
        bbox_yxyx_n23 = tf.concat([bbox_yxyx_n22, ones_yxyx_n21], axis=-1)
        # apply inverse transform
        in_T_out_33 = tf.convert_to_tensor([
            [affine_coeffs[4], affine_coeffs[3], affine_coeffs[5]],
            [affine_coeffs[1], affine_coeffs[0], affine_coeffs[2]],
            [affine_coeffs[7], affine_coeffs[6], 1.],
        ], dtype=tf.float32)
        out_T_in_33 = tf.linalg.inv(in_T_out_33)
        bbox_yxyx_n23 = tf.linalg.matvec(out_T_in_33, bbox_yxyx_n23)
        bbox_size_old_n = bbox_size_old_n * tf.linalg.det(out_T_in_33)
        # put bbox back to regular coordinates
        bbox_yxyx_n22 = bbox_yxyx_n23[..., :-1]
        bbox_yxyx_n4 = tf.reshape(bbox_yxyx_n22, (-1, 4))
        # filter out invisiable / barely visable bboxes
        bbox_yxyx_n4 = tf.minimum(bbox_yxyx_n4, tf.tile(self.output_hw, [2]))
        bbox_yxyx_n4 = tf.maximum(bbox_yxyx_n4, (0., 0., 0., 0.))
        bbox_hw_n2 = bbox_yxyx_n4[:, 2:] - bbox_yxyx_n4[:, :2]
        bbox_size_n = tf.reduce_prod(bbox_hw_n2, axis=-1)
        bbox_visible_ratio_n = tf.math.divide_no_nan(bbox_size_n, bbox_size_old_n)
        bbox_visable_mask_n = (bbox_visible_ratio_n > self.min_bbox_area)
        return bbox_yxyx_n4[bbox_visable_mask_n], bbox_visable_mask_n

    @tf.function
    def __call__(self, data, training=True):
        """
        Args:
            data:       see class definition for data format / customization details
            training:   set to False for validation data to obtain deterministic 
                        center crop behavior
        """
        # get labels from data
        img_hw3 = self.get_image_hw3(data)
        bbox_yxyx_n4 = self.get_bbox_yxyx_n4(data)
        label_n = self.get_label_n(data)
        # generate affine params
        input_hw = tf.cast(img_hw3.shape[:-1], tf.float32)
        if training:
            center_yx, scale_yx = self.get_random_affine(input_hw, bbox_yxyx_n4)
        else:
            center_yx, scale_yx = self.get_center_affine(input_hw)
        affine_coeffs = get_affine_coeffs(self.output_hw, center_yx, scale_yx)
        # apply transformation
        img_hw3 = tfa.image.transform(img_hw3, affine_coeffs, 
            interpolation='BILINEAR', output_shape=tf.cast(self.output_hw, tf.int32))
        bbox_yxyx_n4, bbox_mask_n = self.affine_transform_bbox(affine_coeffs, bbox_yxyx_n4)
        label_n = label_n[bbox_mask_n]
        return self.format_output(img_hw3, bbox_yxyx_n4, label_n)

