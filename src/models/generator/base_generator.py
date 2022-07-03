import tensorflow as tf


class BaseGenerator(tf.keras.Model):
    loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss(self, discriminator_image_evaluation):
        return self.loss_fun(tf.ones_like(discriminator_image_evaluation), discriminator_image_evaluation)

    @staticmethod
    def identity_loss(image, same_image_generated):
        return tf.reduce_mean(tf.abs(image - same_image_generated))
