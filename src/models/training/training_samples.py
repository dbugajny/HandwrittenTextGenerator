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
