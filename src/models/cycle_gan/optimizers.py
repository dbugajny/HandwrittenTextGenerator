import tensorflow as tf
from dataclasses import dataclass


@dataclass
class Optimizers:
    generator_1_optimizer: tf.keras.optimizers.Optimizer
    generator_2_optimizer: tf.keras.optimizers.Optimizer
    discriminator_1_optimizer: tf.keras.optimizers.Optimizer
    discriminator_2_optimizer: tf.keras.optimizers.Optimizer
