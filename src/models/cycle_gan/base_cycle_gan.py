import tensorflow as tf
from dataclasses import dataclass
from models.base_generator import BaseGenerator
from models.base_discriminator import BaseDiscriminator


@dataclass
class Models:
    generator_1: BaseGenerator
    generator_2: BaseGenerator
    discriminator_1: BaseDiscriminator
    discriminator_2: BaseDiscriminator

    @staticmethod
    def cycle_loss(image, cycled_image):
        return tf.reduce_mean(tf.abs(image - cycled_image))
