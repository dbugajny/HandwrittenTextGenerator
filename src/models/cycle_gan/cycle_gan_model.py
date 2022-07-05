import tensorflow as tf
from models.training import cycle_gan_training


class CycleGANModel(tf.keras.Model):
    def __init__(self, architecture):
        super(CycleGANModel, self).__init__()
        self.models = architecture
        self.optimizers = None
        self.losses_fc = None

    def compile(self, optimizers, losses_fc):
        self.optimizers = optimizers
        self.losses_fc = losses_fc

    def train_step(self, image_data):
        image_1, image_2 = image_data

        training_samples, gradient_tape = cycle_gan_training.generate_samples(self.models, image_1, image_2)
        training_losses, gradient_tape = cycle_gan_training.calculate_losses(training_samples, gradient_tape,
                                                                             self.losses_fc)
        training_gradients = cycle_gan_training.calculate_gradients(self.models, training_losses, gradient_tape)
        cycle_gan_training.apply_gradients(self.models, self.optimizers, training_gradients)
