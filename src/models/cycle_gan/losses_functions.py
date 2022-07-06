from dataclasses import dataclass
from typing import Callable


@dataclass
class LossesFunctions:
    generator_loss: Callable
    discriminator_loss: Callable
    cycle_loss: Callable
    identity_loss: Callable
