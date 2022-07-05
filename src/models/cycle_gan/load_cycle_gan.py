import numpy as np
import tensorflow as tf
from models.cycle_gan import cycle_gan_architecture
from models.cycle_gan import cycle_gan_model


def load_cycle_gan_model(path, optimizers):
    architecture = get_cycle_gan_architecture(path)
    zero_gradients, gradient_vars = get_zero_gradients(architecture)
    update_cycle_gan_optimizers(path, optimizers, zero_gradients, gradient_vars)
    return cycle_gan_model.CycleGANModel(architecture, optimizers)


def get_cycle_gan_architecture(path):
    generator_1 = tf.keras.models.load_model(path.joinpath("generator_1"))
    generator_2 = tf.keras.models.load_model(path.joinpath("generator_2"))
    discriminator_1 = tf.keras.models.load_model(path.joinpath("discriminator_1"))
    discriminator_2 = tf.keras.models.load_model(path.joinpath("discriminator_2"))

    architecture = cycle_gan_architecture.CycleGANArchitecture(
        generator_1, generator_2, discriminator_1, discriminator_2
    )

    return architecture


def get_zero_gradients(architecture):
    gradient_vars = (
        architecture.generator_1.trainable_weights,
        architecture.generator_2.trainable_weights,
        architecture.discriminator_1.trainable_weights,
        architecture.discriminator_2.trainable_weights
    )

    zero_gradients = (
        [tf.zeros_like(w) for w in gradient_vars[0]],
        [tf.zeros_like(w) for w in gradient_vars[1]],
        [tf.zeros_like(w) for w in gradient_vars[2]],
        [tf.zeros_like(w) for w in gradient_vars[3]]
    )

    return zero_gradients, gradient_vars


def update_cycle_gan_optimizers(path, optimizers, zero_gradients, gradient_vars):
    optimizers.generator_1_optimizer.apply_gradients(zip(zero_gradients[0], gradient_vars[0]))
    optimizers.generator_2_optimizer.apply_gradients(zip(zero_gradients[1], gradient_vars[1]))
    optimizers.discriminator_1_optimizer.apply_gradients(zip(zero_gradients[2], gradient_vars[2]))
    optimizers.discriminator_2_optimizer.apply_gradients(zip(zero_gradients[3], gradient_vars[3]))

    optimizers_weights = np.load(path.joinpath("optimizers.npy"), allow_pickle=True)
    optimizers.generator_1_optimizer.set_weights(optimizers_weights[0])
    optimizers.generator_2_optimizer.set_weights(optimizers_weights[1])
    optimizers.discriminator_1_optimizer.set_weights(optimizers_weights[2])
    optimizers.discriminator_2_optimizer.set_weights(optimizers_weights[3])
