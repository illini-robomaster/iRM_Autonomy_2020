import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL

# wrapping custom tf logic into Keras Layer
class ImageNormalize(KL.Layer):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        super(ImageNormalize, self).__init__(**kwargs)
        self.mean_3 = tf.constant(mean, tf.float32)
        self.std_3 = tf.constant(std, tf.float32)

    def call(self, image_3hw):
        return (image_3hw - self.mean_3[None, :, None, None]) / self.std_3[None, :, None, None]

# another example for a trainable quadratic function
class Quadratic(KL.Layer):
    def __init__(self, **kwargs):
        super(Quadratic, self).__init__(**kwargs)
        self.a = self.add_weight()
        self.b = self.add_weight()
        self.c = self.add_weight()

    def call(self, x):
        return self.a * x**2 + self.b * x + self.c

# modulerization of network building blocks
class ConvBNReLU(KL.Layer):
    def __init__(self,
        filters,
        kernel_size,
        strides,
        padding='same',
        kernel_initializer='he_normal',
        weight_decay=None,
        **kwargs
    ):
        super(ConvBNReLU, self).__init__(**kwargs)
        self.conv = KL.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=K.regularizers.l2(weight_decay) if weight_decay else None,
            trainable=self.trainable,
            use_bias=False,
            name='conv2d',
        )
        self.bn = KL.BatchNormalization(
            axis=1 if K.backend.image_data_format() == 'channels_first' else 3,
            momentum=0.9,
            epsilon=1e-5,
            trainable=self.trainable,
            name='bn',
        )
        self.relu = KL.ReLU(name='relu')

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

if __name__ == '__main__':
    normalize = ImageNormalize()
    quadratic = Quadratic()
    conv_bn_relu = ConvBNReLU(16, 3, 1)

    x = KL.Input(shape=(3, 224, 224))
    print(x.shape)
    y = normalize(x)
    y = quadratic(y)
    y = conv_bn_relu(y)

    model = K.Model(inputs=x, outputs=y)
    model.summary()
