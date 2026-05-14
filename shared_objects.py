from collections import namedtuple
from enum import Enum


PlayerState = namedtuple('PlayerState', ['id', 'hand', 'table', 'possible_actions'])


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
