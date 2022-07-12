import tensorflow as tf
from models.blocks.down_sample_block import DownSampleBlock


class BetterDiscriminator(tf.keras.Model):
    def __init__(self, filters_list, kernel_size_list):
        super().__init__()

        self.down_sample_blocks = [DownSampleBlock(filters, kernel_size, True) for filters, kernel_size in
                                   zip(filters_list, kernel_size_list)]

        self.conv_1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation='tanh')
        self.conv_2 = tf.keras.layers.Conv2D(128, 3, padding="same", activation='tanh')
        self.out_layer = tf.keras.layers.Conv2D(3, 1, padding="same", activation='tanh')

    def call(self, inputs):
        x = inputs
        for i in range(len(self.down_sample_blocks)):
            x = self.down_sample_blocks[i](x)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.out_layer(x)

        return x
