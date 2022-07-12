from models.blocks.down_sample_block import DownSampleBlock
from models.blocks.up_sample_block import UpSampleBlock
import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(self, filters_list, kernel_size_list):
        super().__init__()

        self.down_sample_blocks = [DownSampleBlock(filters, kernel_size, True) for filters, kernel_size in
                                   zip(filters_list, kernel_size_list)]
        self.up_sample_blocks = [UpSampleBlock(filters, kernel_size, True) for filters, kernel_size in
                                 zip(filters_list, kernel_size_list)]

        self.out_layer = tf.keras.layers.Conv2D(3, 1, padding="same", activation='tanh')

        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        x = inputs
        skips = []
        for i in range(len(self.down_sample_blocks)):
            x = self.down_sample_blocks[i](x)
            skips.append(x)

        for i in reversed(range(len(self.up_sample_blocks))):
            x = self.concat([x, skips[i]])
            x = self.up_sample_blocks[i](x)

        x = self.out_layer(x)
        return x
