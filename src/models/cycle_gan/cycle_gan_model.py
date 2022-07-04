import tensorflow as tf


class CycleGANModel:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

    # def get_model_checkpoint(self, checkpoint_path, max_to_keep):
    #     checkpoint = tf.train.Checkpoint(generator_1=self.models.generator_1,
    #                                      generator_2=self.models.generator_2,
    #                                      discriminator_1=self.models.discriminator_1,
    #                                      discriminator_2=self.models.discriminator_2,
    #                                      generator_1_optimizer=self.optimizers.generator_1_optimizer,
    #                                      generator_2_optimizer=self.optimizers.generator_2_optimizer,
    #                                      discriminator_1_optimizer=self.optimizers.discriminator_1_optimizer,
    #                                      discriminator_2_optimizer=self.optimizers.discriminator_2_optimizer)
    #
    #     checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_path,
    #                                                     max_to_keep=max_to_keep)
    #
    #     return checkpoint, checkpoint_manager
    #
    # @staticmethod
    # def load_last_checkpoint(checkpoint, checkpoint_manager):
    #     checkpoint.restore(checkpoint_manager.latest_checkpoint)