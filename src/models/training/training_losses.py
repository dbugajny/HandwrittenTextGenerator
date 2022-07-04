from dataclasses import dataclass
import tensorflow as tf


@dataclass
class TrainingLosses:
    total_generator_1_loss: tf.Tensor
    total_generator_2_loss: tf.Tensor

    discriminator_1_loss: tf.Tensor
    discriminator_2_loss: tf.Tensor
