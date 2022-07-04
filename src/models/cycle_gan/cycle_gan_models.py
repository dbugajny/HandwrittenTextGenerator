from dataclasses import dataclass
import tensorflow as tf

@dataclass
class CycleGANModels:
    generator_1: tf.keras.Model
    generator_2: tf.keras.Model
    discriminator_1: tf.keras.Model
    discriminator_2: tf.keras.Model

