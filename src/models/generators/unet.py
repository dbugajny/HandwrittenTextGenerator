from models.utils.down_sample_block import DownSampleBlock
from models.utils.up_sample_block import UpSampleBlock
import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.down_sample_blocks = [DownSampleBlock(256, (3, 3), True) for _ in range(8)]
        self.up_sample_blocks = [UpSampleBlock(256, (3, 3), True) for _ in range(8)]

        self.out_layer = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation='tanh')

        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        x = inputs
        skips = []
        for i in range(len(self.down_sample_blocks)):
            x = self.down_sample_blocks[i](x)
            skips.append(x)
