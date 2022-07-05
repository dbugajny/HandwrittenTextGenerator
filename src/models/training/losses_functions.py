from dataclasses import dataclass
import tensorflow as tf


@dataclass
class LossFunctions:
    cycle_loss: tf.Tensor
    discriminator_loss: tf.Tensor
    generator_loss: tf.Tensor
    identity_loss: tf.Tensor
