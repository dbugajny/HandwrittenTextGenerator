import tensorflow as tf


class BaseDiscriminator(tf.keras.Models):
    def __init__(self) -> None:
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __call__(self, *args, **kwargs) -> tf.Tensor:
        pass

    def loss(self, image_evaluation, is_real):
        expected = tf.ones_like(image_evaluation) if is_real else tf.ones_like(image_evaluation)
        return self.loss(expected, image_evaluation)
