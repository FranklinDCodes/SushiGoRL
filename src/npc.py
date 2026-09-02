
from enum import Enum
from src.global_constants import *
import random
import torch


class NPC:

    def __init__(self):
        pass

    def select_action(self, state: PlayerState) -> Action:
        raise NotImplementedError("NPC needs a defined action selection method")

class RandomNPC(NPC):

    def select_action(self, state: PlayerState) -> Action:

        idx = torch.randint(0, len(state.possible_actions), (1,))
        random_choice = state.possible_actions[idx.item()].item()

        return random_choice

class DumperPuddingSafe(NPC):

    """
        A dumpling-focused npc that tries to be pudding safe
        priorities are as follows:
            1. 1 pudding
            2. Dumplings
            3. Maki rolls (3s or 2s)
            4. Everything else
    """

    def select_action(self, state: PlayerState) -> Action:

        # if no pudding
        # check first person table position for pudding
        if state.table[AGENT_TABLE_POS, Card.Pudding.value] == 0:

            # and pudding in hand
            if torch.any(Card.Pudding.value == state.possible_actions):
                return Card.Pudding.value

        # check for dumplings
        if torch.any(Card.Dumpling.value == state.possible_actions):
            return Card.Dumpling.value

        # check for Makis
        if torch.any(Card.Maki_3.value == state.possible_actions):
            return Card.Maki_3.value
        if torch.any(Card.Maki_2.value == state.possible_actions):
            return Card.Maki_2.value

        # else return random choice
        idx = torch.randint(0, len(state.possible_actions), (1,))
        random_choice = state.possible_actions[idx.item()].item()
        return random_choice

class Greedy(NPC):

    """
        A nigiri-focused npc that goes for quick points
        priorities are as follows:
            1. Squid Nigiri
            2. Second Tempura
            3. Salmon Nigiri
            4. Wasabi
            5. Dumplings
            6. Egg Nigiri
            7. First Tempura
            8. Everything else
    """

    def select_action(self, state: PlayerState) -> Action:

        # check for Squid nigiri
        if torch.any(Card.Squid_Nigiri.value == state.possible_actions):
            return Card.Squid_Nigiri.value

        # if one tempura on table
        if state.table[AGENT_TABLE_POS, Card.Tempura.value] % 2 == 1:

            # and tempura to take
            if torch.any(Card.Tempura.value == state.possible_actions):
                return Card.Tempura.value

        # check for Salmon nigiri
        if torch.any(Card.Salmon_Nigiri.value == state.possible_actions):
            return Card.Salmon_Nigiri.value

        # check for Wasabi
        if torch.any(Card.Wasabi.value == state.possible_actions):
            return Card.Wasabi.value

        # check for Dumpling
        if torch.any(Card.Dumpling.value == state.possible_actions):
            return Card.Dumpling.value

        # check for Egg Nigiri
        if torch.any(Card.Egg_Nigiri.value == state.possible_actions):
            return Card.Egg_Nigiri.value

        # check for First tempura
        if torch.any(Card.Tempura.value == state.possible_actions):
            return Card.Tempura.value
        
        idx = torch.randint(0, len(state.possible_actions), (1,))
        random_choice = state.possible_actions[idx.item()].item()

        return random_choice

class Franklin(NPC):

    """
        Franklin
        priorities are as follows:
            0. Squid or salmon if wasabi down
            1. Sashami (early game)
            2. Squid Nigiri
            2B. Egg nirigi if wasabi down
            3. Wasabi (early game)
            4. Salmon Nigiri
            5. Pudding
            5. Tempura
            7. Everything else
    """

    def select_action(self, state: PlayerState) -> Action:

        # if unused wasabi
        unused_wasabi = state.table[AGENT_TABLE_POS, Card.Wasabi.value] > torch.sum(state.table[AGENT_TABLE_POS, Card.Egg_Nigiri_with_Wasabi.value : Card.Squid_Nigiri_with_Wasabi.value + 1])
        if unused_wasabi:

            # and squid nigiri
            if torch.any(Card.Squid_Nigiri.value == state.possible_actions):
                return Card.Squid_Nigiri.value

            # or salmon nigiri
            if torch.any(Card.Salmon_Nigiri.value == state.possible_actions):
                return Card.Salmon_Nigiri.value

        # if sashimi to take
        if torch.any(Card.Sashimi.value == state.possible_actions):

            # and less than two cards on the table (early game grab the first)
            if state.table[AGENT_TABLE_POS, :].sum() < 2:
                return Card.Sashimi.value

            # or an imcomplete set of sashimi on the table
            if state.table[AGENT_TABLE_POS, Card.Sashimi.value] < 3 and state.table[AGENT_TABLE_POS, Card.Sashimi.value] > 0:
                return Card.Sashimi.value

        # if squid nigiri to take
        if torch.any(Card.Squid_Nigiri.value == state.possible_actions):
            return Card.Squid_Nigiri.value

        # unused wasabi and egg nigiri
        if unused_wasabi:
            if torch.any(Card.Egg_Nigiri.value == state.possible_actions):
                return Card.Egg_Nigiri.value

        # if wasabi to take
        if torch.any(Card.Wasabi.value == state.possible_actions):

            # and less than two cards on the table (relatively early game)
            if state.table[AGENT_TABLE_POS, :].sum() < 3:
                return Card.Wasabi.value
        
        # if salmon nigiri to take
        if torch.any(Card.Salmon_Nigiri.value == state.possible_actions):
            return Card.Salmon_Nigiri.value

        # take pudding
        if torch.any(Card.Pudding.value == state.possible_actions):
            return Card.Pudding.value

        idx = torch.randint(0, len(state.possible_actions), (1,))
        random_choice = state.possible_actions[idx.item()].item()

        return random_choice


npcs = {
    "random": RandomNPC,
    "DumperPuddingSafe": DumperPuddingSafe,
    "Greedy": Greedy,
    "Franklin": Franklin
}

def get_npc(name: str, **kwargs):

    return npcs[name](**kwargs)




