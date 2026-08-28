
from collections import namedtuple, deque
from enum import Enum
import random
import torch


# card enum for deck and dealing and table cards
class Card(Enum):
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
    Egg_Nigiri_with_Wasabi = 12
    Salmon_Nigiri_with_Wasabi = 13
    Squid_Nigiri_with_Wasabi = 14


# number of unique cards
# does not count wasabi combos
CARD_NUM = 12

# deck template for reshuffling
DECK_CARDS = [
    *[Card.Tempura for i in range(14)],
    *[Card.Sashimi for i in range(14)],
    *[Card.Dumpling for i in range(14)],
    *[Card.Maki_1 for i in range(6)],
    *[Card.Maki_2 for i in range(12)],
    *[Card.Maki_3 for i in range(8)],
    *[Card.Egg_Nigiri for i in range(5)],
    *[Card.Salmon_Nigiri for i in range(10)],
    *[Card.Squid_Nigiri for i in range(5)],
    *[Card.Pudding for i in range(10)],
    *[Card.Wasabi for i in range(6)],
    *[Card.Chopsticks for i in range(4)]
]

# maps player counts to hand sizes
HAND_SIZES = {
    2: 10,
    3: 9,
    4: 8,
    5: 7
}
POSSIBLE_PLAYER_COUNTS = list(HAND_SIZES.keys())

# the table index that corresponds to the ego agent
AGENT_TABLE_POS = 0


PlayerState = namedtuple('PlayerState', ('id', 'hand', 'table', 'possible_actions'))

Timestep = namedtuple('Timestep',
                        ('state', 'action', 'next_state', 'reward'))

class MemoryBuffer:

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

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
