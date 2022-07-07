from models.utils.down_sample_block import DownSampleBlock
from models.utils.up_sample_block import UpSampleBlock
import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(self, nr_blocks, filters, kernel_size):
        super().__init__()

        self.down_sample_blocks = [DownSampleBlock(filters, kernel_size, True) for _ in range(nr_blocks)]
        self.up_sample_blocks = [UpSampleBlock(filters, kernel_size, True) for _ in range(nr_blocks - 1)]

        self.out_layer = tf.keras.layers.Conv2D(3, kernel_size, padding="same", activation='tanh')

        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        x = inputs
        skips = []
        for i in range(len(self.down_sample_blocks)):
            x = self.down_sample_blocks[i](x)
            skips.append(x)

        for i in reversed(range(len(self.up_sample_blocks))):
            x = self.up_sample_blocks[i](x)
            x = self.concat([x, skips[i]])

        x = self.out_layer(x)
        return x
