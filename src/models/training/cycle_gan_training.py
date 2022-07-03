import tensorflow as tf
from models.training.training_utils import TrainingSamples, TrainingGradients, TrainingLosses, LossesWeights


class CycleGANTraining:
    def __init__(self):
        pass

    def train(self, cycle_gan, image_1, image_2, loss_weights):
        training_samples, gradient_tape = self.generate_samples(cycle_gan, image_1, image_2)
        training_losses, gradient_tape = self.calculate_losses(cycle_gan, training_samples, gradient_tape, loss_weights)
        training_gradients = self.calculate_gradients(cycle_gan, training_losses, gradient_tape)
        self.training_step(cycle_gan, training_gradients)

    @staticmethod
    def generate_samples(cycle_gan, image_1, image_2):
        with tf.GradientTape(persistent=True) as gradient_tape:
            fake_image_2 = cycle_gan.models.generator_1(image_1, training=True)
            cycled_image_1 = cycle_gan.models.generator_2(fake_image_2, training=True)

            fake_image_1 = cycle_gan.models.generator_2(image_2, training=True)
            cycled_image_2 = cycle_gan.models.generator_1(fake_image_1, training=True)

            same_image_generated_1 = cycle_gan.models.generator_2(image_1, training=True)
            same_image_generated_2 = cycle_gan.models.generator_1(image_2, training=True)

            real_image_evaluation_1 = cycle_gan.models.discriminator_1(image_1, training=True)
            real_image_evaluation_2 = cycle_gan.models.discriminator_2(image_2, training=True)

            fake_image_evaluation_1 = cycle_gan.models.discriminator_1(fake_image_1, training=True)
            fake_image_evaluation_2 = cycle_gan.models.discriminator_2(fake_image_2, training=True)

        training_samples = TrainingSamples(
            image_1, image_2,
            fake_image_1, fake_image_2,
            cycled_image_1, cycled_image_2,
            same_image_generated_1, same_image_generated_2,
            real_image_evaluation_1, real_image_evaluation_2,
            fake_image_evaluation_1, fake_image_evaluation_2
        )

        return training_samples, gradient_tape

    @staticmethod
    def calculate_losses(cycle_gan, training_samples, gradient_tape, loss_weights):
        with gradient_tape:
            generator_1_loss = cycle_gan.models.generator_1.loss(training_samples.fake_image_evaluation_2)
            generator_2_loss = cycle_gan.models.generator_2.loss(training_samples.fake_image_evaluation_1)
            generator_1_loss *= loss_weights.generator_loss_weight
            generator_2_loss *= loss_weights.generator_loss_weight

            identity_1_loss = cycle_gan.models.generator_1.identity_loss(training_samples.image_2,
                                                                         training_samples.same_image_generated_2)
            identity_2_loss = cycle_gan.models.generator_2.identity_loss(training_samples.image_1,
                                                                         training_samples.same_image_generated_1)
            identity_1_loss *= loss_weights.identity_loss_weight
            identity_2_loss *= loss_weights.identity_loss_weight

            cycle_1_loss = cycle_gan.cycle_loss(training_samples.image_1, training_samples.cycled_image_1)
            cycle_2_loss = cycle_gan.cycle_loss(training_samples.image_2, training_samples.cycled_image_2)
            cycle_1_loss *= loss_weights.cycle_loss_weight
            cycle_2_loss *= loss_weights.cycle_loss_weight

            total_cycle_loss = cycle_1_loss + cycle_2_loss

            total_generator_1_loss = generator_1_loss + total_cycle_loss + identity_1_loss
            total_generator_2_loss = generator_2_loss + total_cycle_loss + identity_2_loss

            discriminator_1_real_loss = cycle_gan.models.discriminator_1.loss(training_samples.real_image_evaluation_1,
                                                                              is_real=True)
            discriminator_1_fake_loss = cycle_gan.models.discriminator_1.loss(training_samples.fake_image_evaluation_1,
                                                                              is_real=False)

            discriminator_2_real_loss = cycle_gan.models.discriminator_2.loss(training_samples.real_image_evaluation_2,
                                                                              is_real=True)
            discriminator_2_fake_loss = cycle_gan.models.discriminator_2.loss(training_samples.fake_image_evaluation_2,
                                                                              is_real=False)
            discriminator_1_loss = discriminator_1_real_loss + discriminator_1_fake_loss
            discriminator_2_loss = discriminator_2_real_loss + discriminator_2_fake_loss
            discriminator_1_loss *= loss_weights.discriminator_loss_weight
            discriminator_2_loss *= loss_weights.discriminator_loss_weight

        training_losses = TrainingLosses(
            total_generator_1_loss, total_generator_2_loss,
            discriminator_1_loss, discriminator_2_loss
        )

        return training_losses, gradient_tape

    @staticmethod
    @tf.function
    def calculate_gradients(cycle_gan, training_losses, gradient_tape):
        generator_1_gradients = gradient_tape.gradient(training_losses.total_generator_1_loss,
                                                       cycle_gan.models.generator_1.trainable_variables)
        generator_2_gradients = gradient_tape.gradient(training_losses.total_generator_2_loss,
                                                       cycle_gan.models.generator_2.trainable_variables)

        discriminator_1_gradients = gradient_tape.gradient(training_losses.discriminator_1_loss,
                                                           cycle_gan.models.discriminator_1.trainable_variables)
        discriminator_2_gradients = gradient_tape.gradient(training_losses.discriminator_2_loss,
                                                           cycle_gan.models.discriminator_2.trainable_variables)

        training_gradients = TrainingGradients(
            generator_1_gradients, generator_2_gradients,
            discriminator_1_gradients, discriminator_2_gradients
        )
        return training_gradients

    @staticmethod
    @tf.function
    def training_step(cycle_gan, training_gradients):
        generator_1_grads_and_vars = zip(training_gradients.generator_1_gradients,
                                         cycle_gan.models.generator_2.trainable_variables)
        generator_2_grads_and_vars = zip(training_gradients.generator_2_gradients,
                                         cycle_gan.models.generator_2.trainable_variables)
        discriminator_1_grads_and_vars = zip(training_gradients.discriminator_1_gradients,
                                             cycle_gan.models.discriminator_1.trainable_variables)
        discriminator_2_grads_and_vars = zip(training_gradients.discriminator_2_gradients,
                                             cycle_gan.models.discriminator_2.trainable_variables)

        cycle_gan.optimizers.generator_1_optimizer.apply_gradients(generator_1_grads_and_vars)
        cycle_gan.optimizers.generator_2_optimizer.apply_gradients(generator_2_grads_and_vars)
        cycle_gan.optimizers.discriminator_1_optimizer.apply_gradients(discriminator_1_grads_and_vars)
        cycle_gan.optimizers.discriminator_2_optimizer.apply_gradients(discriminator_2_grads_and_vars)
