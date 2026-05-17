
from enum import Enum
from shared_objects import *
import random


class NPC:

    def __init__(self):
        pass

    def select_action(self, state: PlayerState) -> Action:
        pass


class RandomNPC(NPC):

    def select_action(self, state: PlayerState) -> Action:

        return random.choice(state.possible_actions.tolist())
        

npcs = {
    "random": RandomNPC
}

def get_npc(name: str, **kwargs):

    return npcs[name](**kwargs)




