from dataclasses import dataclass
from typing import Callable

import tensorflow as tf


@dataclass
class CycleGANLossesFunctions:
    generator_loss_function: Callable
    discriminator_loss_function: Callable
    cycle_loss_function: Callable
    identity_loss_function: Callable


def cycle_loss(loss_f, weight=1):
    def cycle_loss_f(image, cycled_image):
        loss = loss_f(image, cycled_image)
        return loss * weight

    return cycle_loss_f


def identity_loss(loss_f, weight=1):
    def identity_loss_f(image, same_image_generated):
        loss = loss_f(image, same_image_generated)
        return loss * weight

    return identity_loss_f


def generator_loss(loss_f, weight=1):
    def generator_loss_f(discriminator_image_evaluation):
        loss = loss_f(tf.ones_like(discriminator_image_evaluation), discriminator_image_evaluation)
        return loss * weight

    return generator_loss_f


def discriminator_loss(loss_f, weight=1):
    def discriminator_loss_f(real_image_evaluation, fake_image_evaluation):
        real_loss = loss_f(tf.ones_like(real_image_evaluation), real_image_evaluation)
        fake_loss = loss_f(tf.zeros_like(fake_image_evaluation), fake_image_evaluation)
        return (real_loss + fake_loss) * weight

    return discriminator_loss_f
