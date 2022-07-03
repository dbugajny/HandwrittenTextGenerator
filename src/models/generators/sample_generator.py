import tensorflow as tf
from models.generators.base_generator import BaseGenerator


class SampleGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

        self.conv_1 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")
        self.conv_3 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")

        self.max_pool_1 = tf.keras.layers.MaxPool2D((2, 2))
        self.max_pool_2 = tf.keras.layers.MaxPool2D((2, 2))
        self.max_pool_3 = tf.keras.layers.MaxPool2D((2, 2))

        self.conv_4 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")
        self.conv_5 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")
        self.conv_6 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")

        self.conv_t_1 = tf.keras.layers.Conv2DTranspose(3, (2, 2), strides=(2, 2), padding="same", activation="relu")
        self.conv_t_2 = tf.keras.layers.Conv2DTranspose(3, (2, 2), strides=(2, 2), padding="same", activation="relu")
        self.conv_t_3 = tf.keras.layers.Conv2DTranspose(3, (2, 2), strides=(2, 2), padding="same", activation="relu")

        self.outputs = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)
        x = self.max_pool_2(x)

        x = self.conv_4(x)
        x = self.conv_t_1(x)
        x = self.conv_5(x)
        x = self.conv_t_2(x)
        x = self.conv_6(x)
        x = self.conv_t_3(x)

        return self.outputs(x)
