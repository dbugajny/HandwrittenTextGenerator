from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf


@dataclass
class CycleGANArchitecture:
    generator_1: tf.keras.Model
    generator_2: tf.keras.Model
    discriminator_1: tf.keras.Model
    discriminator_2: tf.keras.Model


def load_cycle_gan_architecture(model_path):
    architecture_path = Path(model_path) / "architecture"
    generator_1 = tf.keras.models.load_model(architecture_path / "generator_1")
    generator_2 = tf.keras.models.load_model(architecture_path / "generator_2")
    discriminator_1 = tf.keras.models.load_model(architecture_path / "discriminator_1")
    discriminator_2 = tf.keras.models.load_model(architecture_path / "discriminator_2")

    return CycleGANArchitecture(generator_1, generator_2, discriminator_1, discriminator_2)


def save_cycle_gan_architecture(model_path, architecture):
    architecture_path = Path(model_path) / "architecture"
    architecture.generator_1.save(architecture_path / "generator_1")
    architecture.generator_2.save(architecture_path / "generator_2")
    architecture.discriminator_1.save(architecture_path / "discriminator_1")
    architecture.discriminator_2.save(architecture_path / "discriminator_2")
