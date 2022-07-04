from dataclasses import dataclass


@dataclass
class LossesWeights:
    generator_loss_weight: float
    identity_loss_weight: float
    cycle_loss_weight: float
    discriminator_loss_weight: float
