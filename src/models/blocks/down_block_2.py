import tensorflow as tf


class DownBlock2(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.max_pooling = tf.keras.layers.MaxPooling2D(padding="same")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.max_pooling(x)
        return x
