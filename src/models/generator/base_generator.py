import tensorflow as tf


class BaseGenerator(tf.keras.Models):
    def __init__(self) -> None:
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __call__(self, *args, **kwargs) -> tf.Tensor:
        pass

    def loss(self, discriminator_image_evaluation):
        return self.loss(tf.ones_like(discriminator_image_evaluation), discriminator_image_evaluation)

    @staticmethod
    def identity_loss(image, same_image_generated):
        return tf.reduce_mean(tf.abs(image - same_image_generated))
