import tensorflow as tf
from models.discriminators.base_discriminator import BaseDiscriminator


class SampleDiscriminator(BaseDiscriminator):
    def __init__(self):
        super().__init__()

        self.conv_1 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")
        self.conv_3 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")

        self.max_pool_1 = tf.keras.layers.MaxPool2D((2, 2))
        self.max_pool_2 = tf.keras.layers.MaxPool2D((2, 2))
        self.max_pool_3 = tf.keras.layers.MaxPool2D((2, 2))

        self.outputs = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)
        x = self.max_pool_2(x)

        return self.outputs(x)
