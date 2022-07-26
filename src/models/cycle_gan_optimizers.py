from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass
class CycleGANOptimizers:
    generator_1_optimizer: tf.keras.optimizers.Optimizer
    generator_2_optimizer: tf.keras.optimizers.Optimizer
    discriminator_1_optimizer: tf.keras.optimizers.Optimizer
    discriminator_2_optimizer: tf.keras.optimizers.Optimizer


def load_cycle_gan_optimizers_weights(model_path, optimizers, architecture):
    model_path = Path(model_path)
    generator_1_optimizer_weights = np.load(model_path / "optimizers_weights" / "generator_1.npy", allow_pickle=True)
    generator_2_optimizer_weights = np.load(model_path / "optimizers_weights" / "generator_2.npy", allow_pickle=True)
    discriminator_1_optimizer_weights = np.load(
        model_path / "optimizers_weights" / "discriminator_1.npy", allow_pickle=True
    )
    discriminator_2_optimizer_weights = np.load(
        model_path / "optimizers_weights" / "discriminator_2.npy", allow_pickle=True
    )

    generator_1_zero_gradients = [tf.zeros_like(w) for w in architecture.generator_1.trainable_weights]
    generator_2_zero_gradients = [tf.zeros_like(w) for w in architecture.generator_2.trainable_weights]
    discriminator_1_zero_gradients = [tf.zeros_like(w) for w in architecture.discriminator_1.trainable_weights]
    discriminator_2_zero_gradients = [tf.zeros_like(w) for w in architecture.discriminator_2.trainable_weights]

    optimizers.generator_1_optimizer.apply_gradients(
        zip(generator_1_zero_gradients, architecture.generator_1.trainable_weights)
    )
    optimizers.generator_2_optimizer.apply_gradients(
        zip(generator_2_zero_gradients, architecture.generator_2.trainable_weights)
    )
    optimizers.discriminator_1_optimizer.apply_gradients(
        zip(discriminator_1_zero_gradients, architecture.discriminator_1.trainable_weights)
    )
    optimizers.discriminator_2_optimizer.apply_gradients(
        zip(discriminator_2_zero_gradients, architecture.discriminator_2.trainable_weights)
    )

    optimizers.generator_1_optimizer.set_weights(generator_1_optimizer_weights)
    optimizers.generator_2_optimizer.set_weights(generator_2_optimizer_weights)
    optimizers.discriminator_1_optimizer.set_weights(discriminator_1_optimizer_weights)
    optimizers.discriminator_2_optimizer.set_weights(discriminator_2_optimizer_weights)


def save_cycle_gan_optimizer_weights(model_path, cycle_gan_optimizers):
    optimizers_weights_path = Path(model_path) / "optimizers_weights"
    optimizers_weights_path.mkdir(exist_ok=True)

    np.save(
        optimizers_weights_path / "generator_1.npy",
        cycle_gan_optimizers.generator_1_optimizer.get_weights()
    )
    np.save(
        optimizers_weights_path / "generator_2.npy",
        cycle_gan_optimizers.generator_2_optimizer.get_weights()
    )
    np.save(
        optimizers_weights_path / "discriminator_1.npy",
        cycle_gan_optimizers.discriminator_1_optimizer.get_weights()
    )
    np.save(
        optimizers_weights_path / "discriminator_2.npy",
        cycle_gan_optimizers.discriminator_2_optimizer.get_weights()
    )
