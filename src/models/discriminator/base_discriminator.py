import tensorflow as tf


class BaseDiscriminator(tf.keras.Models):
    loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss(self, image_evaluation, is_real):
        expected = tf.ones_like(image_evaluation) if is_real else tf.ones_like(image_evaluation)
        return self.loss_fun(expected, image_evaluation)
