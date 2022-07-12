import tensorflow as tf


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, batch_norm=True):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.conv_t = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                                      strides=2, padding="same")
        self.additional_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                                      strides=1, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv_t(inputs)
        x = self.additional_conv(x)
        if self.batch_norm:
            x = self.normalization(x)
        x = self.activation(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters": self.filters, "kernel_size": self.kernel_size, "batch": self.batch_norm}
