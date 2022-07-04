from dataclasses import dataclass
import tensorflow as tf


@dataclass
class TrainingGradients:
    generator_1_gradients: tf.Tensor
    generator_2_gradients: tf.Tensor

    discriminator_1_gradients: tf.Tensor
    discriminator_2_gradients: tf.Tensor
