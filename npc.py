
from enum import Enum
from global_constants import *
import random


class NPC:

    def __init__(self):
        pass

    def select_action(self, state: PlayerState) -> Action:
        raise NotImplementedError("NPC needs a defined action selection method")


class RandomNPC(NPC):

    def select_action(self, state: PlayerState) -> Action:

        return random.choice(state.possible_actions)
        

npcs = {
    "random": RandomNPC
}

def get_npc(name: str, **kwargs):

    return npcs[name](**kwargs)




