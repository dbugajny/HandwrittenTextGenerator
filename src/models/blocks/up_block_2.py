import tensorflow as tf


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.up_sampling = tf.keras.layers.UpSampling2D()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.up_sampling(x)
        return x
