from dataclasses import dataclass
import tensorflow as tf


@dataclass
class TrainingSamples:
    image_1: tf.Tensor
    image_2: tf.Tensor

    fake_image_1: tf.Tensor
    fake_image_2: tf.Tensor

    cycled_image_1: tf.Tensor
    cycled_image_2: tf.Tensor

    same_image_generated_1: tf.Tensor
    same_image_generated_2: tf.Tensor

    real_image_evaluation_1: tf.Tensor
    real_image_evaluation_2: tf.Tensor

    fake_image_evaluation_1: tf.Tensor
    fake_image_evaluation_2: tf.Tensor


@dataclass
class TrainingGradients:
    generator_1_gradients: tf.Tensor
    generator_2_gradients: tf.Tensor

    discriminator_1_gradients: tf.Tensor
    discriminator_2_gradients: tf.Tensor


@dataclass
class TrainingLosses:
    total_generator_1_loss: tf.Tensor
    total_generator_2_loss: tf.Tensor

    discriminator_1_loss: tf.Tensor
    discriminator_2_loss: tf.Tensor


@dataclass
class LossesWeights:
    generator_loss_weight: float
    identity_loss_weight: float
    cycle_loss_weight: float
    discriminator_loss_weight: float
