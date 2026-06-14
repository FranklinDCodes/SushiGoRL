from collections import deque, namedtuple
from enum import Enum
import random
import numpy as np
from global_constants import *



# think about vectorizing further
class Deck(deque):

    def __init__(self, seed=42):

        super(deque, self).__init__()

        random.seed(seed)

        self.reshuffle()

    def reshuffle(self) -> None:

        self.clear()

        # shuffle cards
        deck_cards_copy = DECK_CARDS.copy()
        random.shuffle(deck_cards_copy)

        # add all to deck
        for i in deck_cards_copy:
            self.append(i)

    def deal(self, players: int) -> np.ndarray[np.ndarray[int]]:

        # create empty hands
        hands = np.zeros((players, CARD_NUM))

        # create hands that are card count vectors
        hand_size = HAND_SIZES[players]
        for player in range(players):
            for i in range(hand_size):
                card = self.pop()
                hands[player, card.value] += 1
    
        return np.array(hands)


# class for tracking cards on the table
class Table:

    def __init__(self, player_count: int):

        self.player_count = player_count

        # init table vector
        self.vec = np.zeros((player_count, Card.__len__()), dtype=int)

    def play_cards(self, card_vec: np.ndarray[int]) -> None:

        # card vec is list of cards, not count vector

        np_updated = np.zeros((self.vec.shape[0]), dtype=bool)

        # check if someone laid a nigiri
        np_new_egg_nigiris = (card_vec == Card.Egg_Nigiri.value)
        np_new_salmon_nigiris = (card_vec == Card.Salmon_Nigiri.value)
        np_new_squid_nigiris = (card_vec == Card.Squid_Nigiri.value)
        np_new_nigiri = (np_new_egg_nigiris) | (np_new_salmon_nigiris) | (np_new_squid_nigiris)
        if np.any(np_new_nigiri):

            # check if there are unused wasabi down
            np_unused_wasabi = self.vec[:, Card.Wasabi.value] != np.sum(self.vec[:, Card.Egg_Nigiri_with_Wasabi.value : Card.Salmon_Nigiri_with_Wasabi.value], axis=1)
            if np.any(np_unused_wasabi & np_new_nigiri):

                # # subtract old nigiris
                # self.vec[(np_new_egg_nigiris) & (np_unused_wasabi), Card.Egg_Nigiri.value] -= 1
                # self.vec[(np_new_salmon_nigiris) & (np_unused_wasabi), Card.Salmon_Nigiri.value] -= 1
                # self.vec[(np_new_squid_nigiris) & (np_unused_wasabi), Card.Squid_Nigiri.value] -= 1

                # add new wasabi nigiris
                self.vec[(np_new_egg_nigiris) & (np_unused_wasabi), Card.Egg_Nigiri_with_Wasabi.value] += 1
                self.vec[(np_new_salmon_nigiris) & (np_unused_wasabi), Card.Salmon_Nigiri_with_Wasabi.value] += 1
                self.vec[(np_new_squid_nigiris) & (np_unused_wasabi), Card.Squid_Nigiri_with_Wasabi.value] += 1

                np_updated[(np_new_egg_nigiris | np_new_salmon_nigiris | np_new_squid_nigiris) & (np_unused_wasabi)] = 1

        self.vec[np.where(~np_updated)[0], card_vec[~np_updated]] += 1

    def use_chopsticks(self, player_num: int, card: Card) -> None:

        # check if need to update wasabi sitch
        is_nigiri = (card == Card.Egg_Nigiri.value) or (card == Card.Salmon_Nigiri.value) or (card == Card.Squid_Nigiri.value)
        unused_wasabi = self.vec[player_num, Card.Wasabi.value] != np.sum(self.vec[player_num, Card.Egg_Nigiri_with_Wasabi.value : Card.Salmon_Nigiri_with_Wasabi.value])
        if unused_wasabi and is_nigiri:

            # update with offset that turns it to the wasabi version of the nigiri
            self.vec[player_num, card + 6] += 1

        else:

            # normal card update
            self.vec[player_num, card] += 1

        # remove chopsticks
        self.vec[player_num, Card.Chopsticks.value] -= 1

    def get_player_points(self) -> np.ndarray[int]:

        np_points = np.zeros((self.vec.shape[0]), dtype=int)
        
        # tempura
        np_points += (self.vec[:, Card.Tempura.value] // 2) * 5

        # sashimi
        np_points += (self.vec[:, Card.Sashimi.value] // 3) * 10

        # dumpling
        dumpling_points = np.array([0, 1, 3, 6, 10, 15])
        np_points += dumpling_points[self.vec[:, Card.Dumpling.value]]


        # maki rolls
        np_maki_totals = (self.vec[:, Card.Maki_1.value] + 2 * self.vec[:, Card.Maki_2.value] + 3 * self.vec[:, Card.Maki_3.value])
        top_maki = np.max(np_maki_totals)

        # if there are any maki rolls, split 6 among the maxs
        if top_maki != 0:
            np_points[np_maki_totals == top_maki] += 6 // np.sum(np_maki_totals == top_maki)

        # if there wasn't a tie for first, split the 3 among the runner-ups
        top_maki_tie = np.sum(np_maki_totals == top_maki) > 1
        if not top_maki_tie:
            runnerup_maki = np.sort(np.unique(np_maki_totals))[-2]
            np_points[np_maki_totals == runnerup_maki] += 3 // np.sum(np_maki_totals == runnerup_maki)


        # nigiri
        np_points += self.vec[:, Card.Egg_Nigiri.value]
        np_points += self.vec[:, Card.Salmon_Nigiri.value] * 2
        np_points += self.vec[:, Card.Squid_Nigiri.value] * 3

        # nigiri with wasabi
        np_points += self.vec[:, Card.Egg_Nigiri_with_Wasabi.value] * 3
        np_points += self.vec[:, Card.Salmon_Nigiri_with_Wasabi.value] * 6
        np_points += self.vec[:, Card.Squid_Nigiri_with_Wasabi.value] * 9

        # pudding
        np_points[self.vec[:, Card.Pudding.value] == np.max(self.vec[:, Card.Pudding.value])] += 6 // np.sum(self.vec[:, Card.Pudding.value] == np.max(self.vec[:, Card.Pudding.value]))
        np_points[self.vec[:, Card.Pudding.value] == np.min(self.vec[:, Card.Pudding.value])] -= 6 // np.sum(self.vec[:, Card.Pudding.value] == np.min(self.vec[:, Card.Pudding.value]))

        return np_points

    def get_player_state(self, num: int) -> np.ndarray[int]:
        return self.vec[num, :]

    def clear(self) -> None:

        # init table vector
        self.vec = np.zeros((self.player_count, Card.__len__()), dtype=int)


# class for tracking rotating hands 
class Hands:

    def __init__(self, player_count: int):

        self.player_count = player_count

        # hand vector
        # contains a count of each card 
        self.hand_vec = None # (player, cards)

        # count of cards per hand
        self.cards_left = 0

        # hand position vector
        self.player_to_hand_pos = np.arange((player_count)) # (player)

    def pass_hands(self):
    
        self.player_to_hand_pos = (self.player_to_hand_pos + 1) % (self.player_count)
        
    def __call__(self) -> np.ndarray:

        return self.hand_vec[self.player_to_hand_pos]

    def set_cards(self, hands: np.ndarray[np.ndarray[int]]) -> None:

        self.hand_vec = hands

        self.cards_left = np.sum(hands[0])

    def take(self, cards: np.ndarray[int]) -> None:
        
        # cards is list of cards, not count vector

        # check if move can be made
        if np.any(self.hand_vec[self.player_to_hand_pos, cards] <= 0):

            offending_player = np.arange(self.hand_vec.shape[0])[self.hand_vec[self.player_to_hand_pos, cards] <= 0][0] 
            offending_card = Card(cards[offending_player]).name
            raise ValueError(f"Player_{offending_player} attempted to lay a(n) {offending_card} card when there is not one in their hand.")

        # decrement counts in hands 
        self.hand_vec[self.player_to_hand_pos, cards] -= 1

        self.cards_left -= 1

    def take_with_chopsticks(self, player_num: int, card: int) -> None:

        # check if move can be made
        if self.hand_vec[self.player_to_hand_pos[player_num], card] <= 0:

            offending_card = Card(card).name
            raise ValueError(f"Player_{player_num} attempted to lay a(n) {offending_card} card with chopsticks when there is not one in their hand.")

        # remove card
        self.hand_vec[self.player_to_hand_pos[player_num], card] -= 1

        # add chopsticks
        self.hand_vec[self.player_to_hand_pos[player_num], Card.Chopsticks.value] += 1


class SushiGo:

    def __init__(self, player_count: int, seed: int = 42):

        self.player_count = player_count

        # create deck
        self.deck = Deck(seed)

        # spawn table
        self.table = Table(player_count)

        # deal hands
        self.hands = Hands(player_count)
        self.hands.set_cards(self.deck.deal(player_count))

        self.round_num = 1
        self.round_over = False

    def get_states(self, *player_idx) -> tuple[PlayerState]:

        if len(player_idx) == 0:
            indices = range(self.player_count)
        else:
            indices = player_idx

        l_states = []

        for player_num in indices:

            # get hand
            hand = self.hands()[player_num]

            # table gets rotated each time so that it's at a 0 offset from the agent's POV
            table = self.table.vec[(np.arange(self.table.vec.shape[0]) + player_num) % self.player_count, :]

            # list of actions
            possible_actions = np.arange(hand.shape[0], dtype=int)[hand != 0]

            if table[0, Card.Chopsticks.value] > 0:
                possible_actions = np.concat((possible_actions, np.array([Action.PlayChopsticks.value])))

            # add each hand and table
            l_states.append(PlayerState(player_num, hand, table, possible_actions))

        return l_states
    
    def get_scores(self) -> np.ndarray:
        
        return self.table.get_player_points().tolist()
    
    def play_cards(self, cards: list[int]) -> None:
        
        # remove from hands
        self.hands.take(cards)
        
        # place on table
        self.table.play_cards(cards)

        # check if cards are gone
        if self.hands.cards_left == 0:
            self.round_over = True

    def play_chopsticks(self, player_num: int, card: int) -> None:

        self.hands.take_with_chopsticks(player_num, card)
        self.table.use_chopsticks(player_num, card)

    def pass_hands(self) -> None:
        self.hands.pass_hands()

    def setup_new_round(self, player_count: int) -> None:

        self.player_count = player_count

        self.round_num += 1
        self.round_over = False

        # deal hands
        self.hands = Hands(self.player_count)
        self.hands.set_cards(self.deck.deal(self.player_count))

        # clear table 
        self.table.clear()

    def setup_new_game(self, player_count: int) -> None:

        self.deck.reshuffle()
        self.setup_new_round(player_count)

    def round_is_over(self) -> bool:
        return self.round_over

    def __str__(self):

        COL_WID = 30

        player_hands = []
        player_tables = []

        # iterate through players
        for player_id in range(self.player_count):

            player_hand = []
            player_table = []

            # for each card in hand, append card X count
            for idx, card_count in enumerate(self.hands()[player_id, :].astype(int).tolist()):
                [player_hand.append(Card(idx).name) for i in range(card_count)]

            # for each card on table, append card X count
            for idx, card_count in enumerate(self.table.vec[player_id, :].astype(int).tolist()):
                [player_table.append(Card(idx).name) for i in range(card_count)]

            player_hands.append(player_hand)
            player_tables.append(player_table)

        string = ''.join([("\033[93mPlayer_" + str(i) + "\033[00m").ljust(COL_WID+10) for i in range(self.player_count)]) + '\n\n'

        # print hands
        string += "\033[34mHANDS\033[00m\n"
        for card_idx in range(len(player_hands[0])):
            string += ''.join([(player_hands[player_idx][card_idx]).ljust(COL_WID) for player_idx in range(self.player_count)]) + '\n'

        string += "\n\033[34mTABLE\033[00m\n"

        # print tables
        for card_idx in range(len(player_tables[0])):
            string += ''.join([(player_tables[player_idx][card_idx]).ljust(COL_WID) for player_idx in range(self.player_count)]) + '\n'

        return string + '\n'

