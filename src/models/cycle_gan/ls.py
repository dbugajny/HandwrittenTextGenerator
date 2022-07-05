import tensorflow as tf


def cycle_loss(image, cycled_image):
    return tf.reduce_mean(tf.abs(image - cycled_image))


def discriminator_loss(image_evaluation, is_real):
    loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    expected = tf.ones_like(image_evaluation) if is_real else tf.ones_like(image_evaluation)
    return loss_fun(expected, image_evaluation)


def generator_loss(discriminator_image_evaluation):
    loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_fun(tf.ones_like(discriminator_image_evaluation), discriminator_image_evaluation)


def identity_loss(image, same_image_generated):
    return tf.reduce_mean(tf.abs(image - same_image_generated))