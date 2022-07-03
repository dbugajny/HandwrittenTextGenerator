from dataclasses import dataclass
from models.generators.base_generator import BaseGenerator
from models.discriminators.base_discriminator import BaseDiscriminator


@dataclass
class CycleGANModels:
    generator_1: BaseGenerator
    generator_2: BaseGenerator
    discriminator_1: BaseDiscriminator
    discriminator_2: BaseDiscriminator

