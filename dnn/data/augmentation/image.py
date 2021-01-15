import tensorflow as tf

class ImageAugmentor(tf.Module):
    '''
    Note that only image is augmented so label are kept same, therefore only x_train is needed

    Input format:
    x_trian of shape h*w*3
    
    Output format:
    x_train of shape h*w*3
    '''

    def __init__(self,
        hue=0.1,
        sat=(1,3),
        constrast=(0.7, 1.0),
        brightness=0.2):
        '''
        Args:
            hue:        max delta for random_hue, range [0, 0.5]
            sat:        [lower, upper] bound for saturation factor
            constrast:  [lower, upper] bound for contrast factor
            brightness: max delta for random brightness, must be none negative
        '''
        super(ImageAugmentor, self).__init__()
        self.hue = hue
        self.sat = sat
        self.contrast = constrast
        self.brightness = brightness

    @tf.function
    def __call__(self, image, training=True):
        """
        Args:
            data:       see class definition for data format / customization details
            training:   set to False for validation data for no image augmentation
        """
        if training:
            image = tf.image.random_saturation(image, self.sat[0], self.sat[1])
            image = tf.image.random_hue(image, self.hue)
            image = tf.image.random_contrast(image, self.contrast[0], self.contrast[1])
            image = tf.image.random_brightness(image, self.brightness)
        return image