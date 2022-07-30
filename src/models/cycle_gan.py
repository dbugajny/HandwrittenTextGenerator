import tensorflow as tf


class CycleGAN(tf.keras.Model):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.optimizers = None
        self.losses_functions = None

    def compile(self, optimizers, losses_functions):
        super().compile()
        self.optimizers = optimizers
        self.losses_functions = losses_functions

    @tf.function
    def train_step(self, image_data):
        image_1, image_2 = image_data

        with tf.GradientTape(persistent=True) as gradient_tape:
            fake_image_2 = self.architecture.generator_1(image_1, training=True)
            fake_image_1 = self.architecture.generator_2(image_2, training=True)

            cycled_image_1 = self.architecture.generator_2(fake_image_2, training=True)
            cycled_image_2 = self.architecture.generator_1(fake_image_1, training=True)

            same_image_generated_1 = self.architecture.generator_2(image_1, training=True)
            same_image_generated_2 = self.architecture.generator_1(image_2, training=True)

            real_image_evaluation_1 = self.architecture.discriminator_1(image_1, training=True)
            real_image_evaluation_2 = self.architecture.discriminator_2(image_2, training=True)

            fake_image_evaluation_1 = self.architecture.discriminator_1(fake_image_1, training=True)
            fake_image_evaluation_2 = self.architecture.discriminator_2(fake_image_2, training=True)

            generator_1_loss = self.losses_functions.generator_loss_function(fake_image_evaluation_2)
            generator_2_loss = self.losses_functions.generator_loss_function(fake_image_evaluation_1)

            identity_1_loss = self.losses_functions.identity_loss_function(image_2, same_image_generated_2)
            identity_2_loss = self.losses_functions.identity_loss_function(image_1, same_image_generated_1)

            cycle_1_loss = self.losses_functions.cycle_loss_function(image_2, cycled_image_2)
            cycle_2_loss = self.losses_functions.cycle_loss_function(image_1, cycled_image_1)
            total_cycle_loss = cycle_1_loss + cycle_2_loss

            total_generator_1_loss = generator_1_loss + cycle_1_loss + identity_1_loss
            total_generator_2_loss = generator_2_loss + cycle_2_loss + identity_2_loss

            discriminator_1_loss = self.losses_functions.discriminator_loss_function(
                real_image_evaluation_1, fake_image_evaluation_1
            )
            discriminator_2_loss = self.losses_functions.discriminator_loss_function(
                real_image_evaluation_2, fake_image_evaluation_2
            )

        generator_1_gradients = gradient_tape.gradient(
            total_generator_1_loss, self.architecture.generator_1.trainable_variables
        )
        generator_2_gradients = gradient_tape.gradient(
            total_generator_2_loss, self.architecture.generator_2.trainable_variables
        )

        discriminator_1_gradients = gradient_tape.gradient(
            discriminator_1_loss, self.architecture.discriminator_1.trainable_variables
        )
        discriminator_2_gradients = gradient_tape.gradient(
            discriminator_2_loss, self.architecture.discriminator_2.trainable_variables
        )

        generator_1_grads_and_vars = zip(generator_1_gradients, self.architecture.generator_1.trainable_variables)
        generator_2_grads_and_vars = zip(generator_2_gradients, self.architecture.generator_2.trainable_variables)
        discriminator_1_grads_and_vars = zip(
            discriminator_1_gradients, self.architecture.discriminator_1.trainable_variables
        )
        discriminator_2_grads_and_vars = zip(
            discriminator_2_gradients, self.architecture.discriminator_2.trainable_variables
        )

        self.optimizers.generator_1_optimizer.apply_gradients(generator_1_grads_and_vars)
        self.optimizers.generator_2_optimizer.apply_gradients(generator_2_grads_and_vars)
        self.optimizers.discriminator_1_optimizer.apply_gradients(discriminator_1_grads_and_vars)
        self.optimizers.discriminator_2_optimizer.apply_gradients(discriminator_2_grads_and_vars)

        return {
            "total_generator_1_loss": total_generator_1_loss,
            "total_generator_2_loss": total_generator_2_loss,
            "discriminator_1_loss": discriminator_1_loss,
            "discriminator_2_loss": discriminator_2_loss,
        }
