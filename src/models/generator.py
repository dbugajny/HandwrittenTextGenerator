import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters_1, filters_2):
        super().__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=filters_1, kernel_size=(3, 3), padding="same")
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters_2, kernel_size=(3, 3), padding="same")
        self.normalization_1 = tf.keras.layers.BatchNormalization()
        self.normalization_2 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.normalization_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.activation(x)
        return x


class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_block_1 = ConvBlock(128, 128)
        self.conv_block_2 = ConvBlock(256, 256)
        self.conv_block_3 = ConvBlock(512, 512)
        self.conv_block_4 = ConvBlock(1024, 1024)
        self.conv_block_5 = ConvBlock(512, 512)
        self.conv_block_6 = ConvBlock(256, 256)
        self.conv_block_7 = ConvBlock(128, 128)
        self.conv_out = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding="same", activation="tanh")

        self.conv_down_sampling_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=2, padding="same")
        self.conv_down_sampling_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=2, padding="same")
        self.conv_down_sampling_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=2, padding="same")

        self.conv_up_sampling_1 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(1, 1), strides=2,
                                                                  padding='same')
        self.conv_up_sampling_2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(1, 1), strides=2,
                                                                  padding='same')
        self.conv_up_sampling_3 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(1, 1), strides=2,
                                                                  padding='same')

    def call(self, inputs):
        x_1 = self.conv_block_1(inputs)
        x_2 = self.conv_down_sampling_1(x_1)
        x_2 = self.conv_block_2(x_2)
        x_3 = self.conv_down_sampling_2(x_2)
        x_3 = self.conv_block_3(x_3)
        x = self.conv_down_sampling_3(x_3)
        x = self.conv_block_4(x)
        x = self.conv_up_sampling_1(x)

        x = self.conv_block_5(tf.keras.layers.Concatenate()([x, x_3]))
        x = self.conv_up_sampling_2(x)
        x = self.conv_block_6(tf.keras.layers.Concatenate()([x, x_2]))
        x = self.conv_up_sampling_3(x)
        x = self.conv_block_7(x)
        x = self.conv_out(x)

        return x

    def summary(self):
        x = tf.keras.layers.Input(shape=(160, 160, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()
