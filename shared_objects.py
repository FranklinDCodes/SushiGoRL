from collections import namedtuple, deque
from enum import Enum
import random


PlayerState = namedtuple('PlayerState', ['id', 'hand', 'table', 'possible_actions'])

Timestep = namedtuple('Timestep',
                        ('state', 'action', 'next_state', 'reward'))

class MemoryBuffer:

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Timestep(*args))

    def sample(self, batch_size: int) -> Timestep:
        samples = random.sample(self.memory, batch_size)
        batched = Timestep(*zip(*samples))
        return batched

    def __len__(self) -> int:
        return len(self.memory)

# possible actions
class Action(Enum):
    Tempura = 0
    Sashimi = 1
    Dumpling = 2
    Maki_1 = 3
    Maki_2 = 4
    Maki_3 = 5
    Egg_Nigiri = 6
    Salmon_Nigiri = 7
    Squid_Nigiri = 8
    Pudding = 9
    Wasabi = 10
    Chopsticks = 11
    PlayChopsticks = 12
