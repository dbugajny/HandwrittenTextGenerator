import tensorflow as tf


class ConvBlock2(tf.keras.layers.Layer):
    def __init__(self, filters_1, filters_2):
        super().__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=filters_1, kernel_size=(3, 3), padding="same")
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters_2, kernel_size=(3, 3), padding="same")
        self.normalization_1 = tf.keras.layers.BatchNormalization()
        self.normalization_2 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.normalization_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.activation(x)
        return x


class Disc2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_block_1 = ConvBlock2(512, 512)
        self.conv_block_2 = ConvBlock2(256, 256)
        self.conv_block_3 = ConvBlock2(128, 128)
        self.conv_block_1 = ConvBlock2(64, 64)

        self.conv_down_sampling_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=2, padding="same")
        self.conv_down_sampling_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=2, padding="same")
        self.conv_down_sampling_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=2, padding="same")
        self.conv_down_sampling_4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=2, padding="same")

        self.conv_additional = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")
        self.conv_out = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same")

        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU()
cd
    def call(self, inputs):
        x = self.conv_block_1(inputs)
        x = self.conv_down_sampling_1(x)
        x = self.conv_block_2(x)
        x = self.conv_down_sampling_2(x)
        x = self.conv_block_3(x)
        x = self.conv_down_sampling_3(x)
        x = self.conv_block_4(x)
        x = self.conv_down_sampling_4(x)
        x = self.conv_additional(x)
        x = self.normalization(x)
        x = self.conv_out(x)
        x = self.activation(x)

        return x

    def summary(self):
        x = tf.keras.layers.Input(shape=(160, 160, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()
