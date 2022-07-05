import tensorflow as tf
from models.training import samples, losses, gradients, losses_functions


def generate_samples(models, image_1, image_2):
    with tf.GradientTape(persistent=True) as gradient_tape:
        fake_image_2 = models.generator_1(image_1, training=True)
        cycled_image_1 = models.generator_2(fake_image_2, training=True)

        fake_image_1 = models.generator_2(image_2, training=True)
        cycled_image_2 = models.generator_1(fake_image_1, training=True)

        same_image_generated_1 = models.generator_2(image_1, training=True)
        same_image_generated_2 = models.generator_1(image_2, training=True)

        real_image_evaluation_1 = models.discriminator_1(image_1, training=True)
        real_image_evaluation_2 = models.discriminator_2(image_2, training=True)

        fake_image_evaluation_1 = models.discriminator_1(fake_image_1, training=True)
        fake_image_evaluation_2 = models.discriminator_2(fake_image_2, training=True)

    training_samples = samples.Samples(
        image_1, image_2,
        fake_image_1, fake_image_2,
        cycled_image_1, cycled_image_2,
        same_image_generated_1, same_image_generated_2,
        real_image_evaluation_1, real_image_evaluation_2,
        fake_image_evaluation_1, fake_image_evaluation_2
    )

    return training_samples, gradient_tape


def calculate_losses(training_samples, gradient_tape, losses_fc):
    with gradient_tape:
        generator_1_loss = losses_fc.generator_loss(training_samples.fake_image_evaluation_2)
        generator_2_loss = losses_fc.generator_loss(training_samples.fake_image_evaluation_1)

        identity_1_loss = losses_fc.identity_loss(training_samples.image_2, training_samples.same_image_generated_2)
        identity_2_loss = losses_fc.identity_loss(training_samples.image_1, training_samples.same_image_generated_1)

        cycle_1_loss = losses_fc.cycle_loss(training_samples.image_1, training_samples.cycled_image_1)
        cycle_2_loss = losses_fc.cycle_loss(training_samples.image_2, training_samples.cycled_image_2)
        total_cycle_loss = cycle_1_loss + cycle_2_loss

        total_generator_1_loss = generator_1_loss + total_cycle_loss + identity_1_loss
        total_generator_2_loss = generator_2_loss + total_cycle_loss + identity_2_loss

        discriminator_1_real_loss = losses_fc.discriminator_loss(training_samples.real_image_evaluation_1, True)
        discriminator_1_fake_loss = losses_fc.discriminator_loss(training_samples.fake_image_evaluation_1, False)
        discriminator_2_real_loss = losses_fc.discriminator_loss(training_samples.real_image_evaluation_2, True)
        discriminator_2_fake_loss = losses_fc.discriminator_loss(training_samples.fake_image_evaluation_2, False)
        discriminator_1_loss = discriminator_1_real_loss + discriminator_1_fake_loss
        discriminator_2_loss = discriminator_2_real_loss + discriminator_2_fake_loss

    training_losses = losses.Losses(
        total_generator_1_loss, total_generator_2_loss,
        discriminator_1_loss, discriminator_2_loss
    )

    return training_losses, gradient_tape


def calculate_gradients(models, training_losses, gradient_tape):
    generator_1_gradients = gradient_tape.gradient(training_losses.total_generator_1_loss,
                                                   models.generator_1.trainable_variables)
    generator_2_gradients = gradient_tape.gradient(training_losses.total_generator_2_loss,
                                                   models.generator_2.trainable_variables)

    discriminator_1_gradients = gradient_tape.gradient(training_losses.discriminator_1_loss,
                                                       models.discriminator_1.trainable_variables)
    discriminator_2_gradients = gradient_tape.gradient(training_losses.discriminator_2_loss,
                                                       models.discriminator_2.trainable_variables)

    training_gradients = gradients.Gradients(
        generator_1_gradients, generator_2_gradients,
        discriminator_1_gradients, discriminator_2_gradients
    )
    return training_gradients


def apply_gradients(models, optimizers, training_gradients):
    generator_1_grads_and_vars = zip(training_gradients.generator_1_gradients,
                                     models.generator_2.trainable_variables)
    generator_2_grads_and_vars = zip(training_gradients.generator_2_gradients,
                                     models.generator_2.trainable_variables)
    discriminator_1_grads_and_vars = zip(training_gradients.discriminator_1_gradients,
                                         models.discriminator_1.trainable_variables)
    discriminator_2_grads_and_vars = zip(training_gradients.discriminator_2_gradients,
                                         models.discriminator_2.trainable_variables)

    optimizers.generator_1_optimizer.apply_gradients(generator_1_grads_and_vars)
    optimizers.generator_2_optimizer.apply_gradients(generator_2_grads_and_vars)
    optimizers.discriminator_1_optimizer.apply_gradients(discriminator_1_grads_and_vars)
    optimizers.discriminator_2_optimizer.apply_gradients(discriminator_2_grads_and_vars)
