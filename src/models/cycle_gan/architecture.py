from dataclasses import dataclass
import tensorflow as tf


@dataclass
class Architecture:
    generator_1: tf.keras.Model
    generator_2: tf.keras.Model
    discriminator_1: tf.keras.Model
    discriminator_2: tf.keras.Model


def load_architecture(model_path):
    generator_1 = tf.keras.models.load_model(model_path / "architecture" / "generator_1")
    generator_2 = tf.keras.models.load_model(model_path / "architecture" / "generator_2")
    discriminator_1 = tf.keras.models.load_model(model_path / "architecture" / "discriminator_1")
    discriminator_2 = tf.keras.models.load_model(model_path / "architecture" / "discriminator_2")

    return Architecture(generator_1, generator_2, discriminator_1, discriminator_2)


def save_architecture(model_path, architecture):
    architecture.generator_1.save(model_path / "architecture" / "generator_1")
    architecture.generator_2.save(model_path / "architecture" / "generator_1")
    architecture.discriminator_1.save(model_path / "architecture" / "generator_1")
    architecture.discriminator_2.save(model_path / "architecture" / "generator_1")
