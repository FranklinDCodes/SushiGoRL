
from collections import deque, namedtuple
from enum import Enum
import torch
from global_constants import *


class Deck(deque):

    def __init__(self, seed: int = 42):

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

    def deal(self, players: int, device: any) -> torch.Tensor:

        # create empty hands
        hands = torch.zeros((players, CARD_NUM), device=device)

        # create hands that are card count vectors
        hand_size = HAND_SIZES[players]
        for player in range(players):
            for i in range(hand_size):
                card = self.pop()
                hands[player, card.value] += 1
    
        return hands


# class for tracking cards on the table
class Table:

    def __init__(self, device: any = 'cpu'):

        self.player_count = None

        # init table vector
        self.vec = None

        self.device = device

    def play_cards(self, card_vec: torch.Tensor) -> None:

        # card vec is list of cards, not count vector

        t_updated = torch.zeros((self.vec.shape[0]), dtype=bool, device=self.device)

        # check if someone laid a nigiri
        t_new_egg_nigiris = (card_vec == Card.Egg_Nigiri.value)
        t_new_salmon_nigiris = (card_vec == Card.Salmon_Nigiri.value)
        t_new_squid_nigiris = (card_vec == Card.Squid_Nigiri.value)
        t_new_nigiri = (t_new_egg_nigiris) | (t_new_salmon_nigiris) | (t_new_squid_nigiris)
        if torch.any(t_new_nigiri):

            # check if there are unused wasabi down
            t_unused_wasabi = self.vec[:, Card.Wasabi.value] > torch.sum(self.vec[:, Card.Egg_Nigiri_with_Wasabi.value : Card.Squid_Nigiri_with_Wasabi.value + 1], dim=1)
            if torch.any(t_unused_wasabi & t_new_nigiri):

                # add new wasabi nigiris
                self.vec[(t_new_egg_nigiris) & (t_unused_wasabi), Card.Egg_Nigiri_with_Wasabi.value] += 1
                self.vec[(t_new_salmon_nigiris) & (t_unused_wasabi), Card.Salmon_Nigiri_with_Wasabi.value] += 1
                self.vec[(t_new_squid_nigiris) & (t_unused_wasabi), Card.Squid_Nigiri_with_Wasabi.value] += 1

                t_updated[(t_new_egg_nigiris | t_new_salmon_nigiris | t_new_squid_nigiris) & (t_unused_wasabi)] = 1

        self.vec[torch.where(~t_updated)[0], card_vec[~t_updated]] += 1

    def use_chopsticks(self, player_num: int, card: Card) -> None:

        # check if need to update wasabi sitch
        is_nigiri = (card == Card.Egg_Nigiri.value) or (card == Card.Salmon_Nigiri.value) or (card == Card.Squid_Nigiri.value)
        unused_wasabi = self.vec[player_num, Card.Wasabi.value] != torch.sum(self.vec[player_num, Card.Egg_Nigiri_with_Wasabi.value : Card.Salmon_Nigiri_with_Wasabi.value])
        if unused_wasabi and is_nigiri:

            # update with offset that turns it to the wasabi version of the nigiri
            self.vec[player_num, card + 6] += 1

        else:

            # normal card update
            self.vec[player_num, card] += 1

        # remove chopsticks
        self.vec[player_num, Card.Chopsticks.value] -= 1

    def get_player_points(self) -> torch.Tensor:

        t_points = torch.zeros((self.vec.shape[0]), dtype=int, device=self.device)
        
        # tempura
        t_points += (self.vec[:, Card.Tempura.value] // 2) * 5

        # sashimi
        t_points += (self.vec[:, Card.Sashimi.value] // 3) * 10

        # dumpling
        # extra 15s built in, in-case player has up to 10 dumplings
        dumpling_points = torch.tensor([0, 1, 3, 6, 10, 15, 15, 15, 15, 15], device=self.device)
        t_points += dumpling_points[self.vec[:, Card.Dumpling.value]]

        # maki rolls
        t_maki_totals = (self.vec[:, Card.Maki_1.value] + 2 * self.vec[:, Card.Maki_2.value] + 3 * self.vec[:, Card.Maki_3.value])
        top_maki = torch.max(t_maki_totals)

        # if there are any maki rolls, split 6 among the maxs
        if top_maki != 0:
            t_points[t_maki_totals == top_maki] += 6 // torch.sum(t_maki_totals == top_maki)

        # if there wasn't a tie for first, split the 3 among the runner-ups
        top_maki_tie = torch.sum(t_maki_totals == top_maki) > 1
        if not top_maki_tie:
            runnerup_maki_count = torch.sort(torch.unique(t_maki_totals)).values[-2]

            # make sure runnerup maki is not 0
            if runnerup_maki_count != 0:
                t_points[t_maki_totals == runnerup_maki_count] += 3 // torch.sum(t_maki_totals == runnerup_maki_count)

        # nigiri
        t_points += self.vec[:, Card.Egg_Nigiri.value]
        t_points += self.vec[:, Card.Salmon_Nigiri.value] * 2
        t_points += self.vec[:, Card.Squid_Nigiri.value] * 3

        # nigiri with wasabi
        t_points += self.vec[:, Card.Egg_Nigiri_with_Wasabi.value] * 3
        t_points += self.vec[:, Card.Salmon_Nigiri_with_Wasabi.value] * 6
        t_points += self.vec[:, Card.Squid_Nigiri_with_Wasabi.value] * 9

        # pudding
        t_points[self.vec[:, Card.Pudding.value] == torch.max(self.vec[:, Card.Pudding.value])] += 6 // torch.sum(self.vec[:, Card.Pudding.value] == torch.max(self.vec[:, Card.Pudding.value]))
        t_points[self.vec[:, Card.Pudding.value] == torch.min(self.vec[:, Card.Pudding.value])] -= 6 // torch.sum(self.vec[:, Card.Pudding.value] == torch.min(self.vec[:, Card.Pudding.value]))

        return t_points

    def get_confirmed_player_points(self) -> torch.Tensor:

        t_points = torch.zeros((self.vec.shape[0]), dtype=int, device=self.device)
        
        # tempura
        t_points += (self.vec[:, Card.Tempura.value] // 2) * 5

        # sashimi
        t_points += (self.vec[:, Card.Sashimi.value] // 3) * 10

        # dumpling
        # extra 15s built in, in-case player has up to 10 dumplings
        dumpling_points = torch.tensor([0, 1, 3, 6, 10, 15, 15, 15, 15, 15], device=self.device)
        t_points += dumpling_points[self.vec[:, Card.Dumpling.value]]

        # nigiri
        t_points += self.vec[:, Card.Egg_Nigiri.value]
        t_points += self.vec[:, Card.Salmon_Nigiri.value] * 2
        t_points += self.vec[:, Card.Squid_Nigiri.value] * 3

        # nigiri with wasabi
        t_points += self.vec[:, Card.Egg_Nigiri_with_Wasabi.value] * 3
        t_points += self.vec[:, Card.Salmon_Nigiri_with_Wasabi.value] * 6
        t_points += self.vec[:, Card.Squid_Nigiri_with_Wasabi.value] * 9

        return t_points

    def get_player_state(self, num: int) -> torch.Tensor:
        return self.vec[num, :]

    def setup(self, player_count: int) -> None:
        self.player_count = player_count
        self.vec = torch.zeros((player_count, Card.__len__()), dtype=int, device=self.device)


# class for tracking rotating hands 
class Hands:

    # this class needs to be like the only place where there is a player seating mapping, not in main

    def __init__(self, player_count: int, device: any = 'cpu'):

        self.player_count = player_count

        # hand vector
        # contains a count of each card 
        self.hand_vec = None # (player, cards)

        # count of cards per hand
        self.cards_left = 0

        self.device = device

        # hand position vector
        self.player_to_hand_pos = torch.arange((player_count), device=self.device) # (player)

    def pass_hands(self) -> None:
    
        self.player_to_hand_pos = (self.player_to_hand_pos + 1) % (self.player_count)
        
    def __call__(self) -> torch.Tensor:

        return self.hand_vec[self.player_to_hand_pos]

    def set_cards(self, hands: torch.Tensor) -> None:

        self.hand_vec = hands

        self.cards_left = torch.sum(hands[0])

    def take(self, cards: torch.Tensor) -> None:
        
        # cards is list of cards, not count vector

        # check if move can be made
        if torch.any(self.hand_vec[self.player_to_hand_pos, cards] <= 0):

            offending_player = torch.arange(self.hand_vec.shape[0], device=self.device)[self.hand_vec[self.player_to_hand_pos, cards] <= 0][0] 
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

    def __init__(self, seed: int = 42, device: any = "cpu", parallelize_count: int = 1):

        self.player_count = None

        # create deck
        self.deck = Deck(seed)

        # spawn table
        self.table = Table(device)

        # init hands
        self.hands = None

        self.round_num = 0
        self.round_over = False

        self.device = device

        # count of games to run simultaneously in parallel
        self.parallelize_count = parallelize_count

        # game score tracker
        self._past_round_scores = list()

    def get_states(self, *player_idx) -> tuple[PlayerState]:

        if len(player_idx) == 0:
            indices = range(self.player_count)
        else:
            indices = player_idx

        l_states = []

        # convert indices 

        for player_num in indices:

            # get hand
            hand = self.hands()[player_num]

            # table gets rotated each time so that it's at a 0 offset from the agent's POV
            table = self.table.vec[(torch.arange(self.table.vec.shape[0], device=self.device) + player_num) % self.player_count, :]

            # list of actions
            possible_actions = self.get_possible_actions(hand, table)

            # add each hand and table
            l_states.append(PlayerState(player_num, hand, table, possible_actions))

        # unpack if only 1
        if len(l_states) == 1:
            return l_states[0]
        else:
            return l_states

    def get_possible_actions(self, hand: torch.Tensor, table: torch.Tensor) -> torch.Tensor:

        # list of actions
        possible_actions = torch.arange(hand.shape[0], dtype=int, device=self.device)[hand != 0]

        if (table[AGENT_TABLE_POS, Card.Chopsticks.value] > 0) and (torch.sum(hand) > 1):
            possible_actions = torch.concatenate((possible_actions, torch.tensor([Action.PlayChopsticks.value], device=self.device)))

        return possible_actions
    
    def get_agent_round_score(self) -> list:
        
        return self.table.get_player_points().cpu().numpy().tolist()[AGENT_TABLE_POS]

    def get_round_scores(self) -> list:
        
        return self.table.get_player_points()

    def get_confirmed_round_scores(self) -> list:
        
        return self.table.get_confirmed_player_points()

    def get_game_scores(self) -> torch.tensor:

        return sum(self._past_round_scores)
    
    def play_cards(self, cards: torch.Tensor) -> None:
        
        # remove from hands
        self.hands.take(cards)
        
        # place on table
        self.table.play_cards(cards)

        # check if cards are gone
        if self.hands.cards_left == 0:
            self.round_over = True

    def play_chopsticks(self, player_num: int, card: int) -> None:

        # function returns state that is left for second choice with the chopsticks

        self.hands.take_with_chopsticks(player_num, card)
        self.table.use_chopsticks(player_num, card)

        resulting_state = self.get_states(player_num)
        resulting_state.hand[Card.Chopsticks.value] -= 1

        # create modified resulting state for getting choice where chopsticks are not in the hand yet
        possible_actions = self.get_possible_actions(resulting_state.hand, resulting_state.table)
        state_for_second_choice = PlayerState(
            player_num,
            resulting_state.hand,
            resulting_state.table,
            torch.Tensor([i for i in possible_actions if i != Action.PlayChopsticks.value], device=self.device)
            
        )

        return state_for_second_choice

    def pass_hands(self) -> None:
        self.hands.pass_hands()

    def setup_new_round(self, player_count: int) -> None:

        # save away scores if not resetting
        if self.round_num != 0:
            self._past_round_scores.append(self.get_round_scores())

        self.player_count = player_count

        self.round_num += 1
        self.round_over = False

        # deal hands
        self.hands = Hands(self.player_count, device=self.device)
        self.hands.set_cards(self.deck.deal(self.player_count, self.device))

        # clear table
        self.table.setup(player_count)

    def setup_new_game(self, player_count: int) -> None:

        self.round_num = 0
        self.deck.reshuffle()
        self.setup_new_round(player_count)
        self._past_round_scores = list()

    def end_game(self) -> None:

        # add the last round to history
        self._past_round_scores.append(self.get_round_scores())

    def round_is_over(self) -> bool:
        return self.round_over

    def __str__(self, print_hands: bool = True, rotate_players_about: any = None, l_player_names: list = None):

        # if any number is passed for rotate_players_about
        # then that number player will be displayed as first (far left col)

        l_player_idx = list(range(self.player_count))

        if rotate_players_about is not None:

            l_player_idx = [(i + rotate_players_about) % self.player_count for i in l_player_idx]

        player_hands = []
        player_tables = []

        # iterate through players
        for player_id in range(self.player_count):

            player_hand = []
            player_table = []

            # for each card in hand, append card X count
            for idx, card_count in enumerate(self.hands()[player_id, :].to(int).cpu().numpy().tolist()):
                [player_hand.append(Card(idx).name) for i in range(card_count)]

            # for each card on table, append card X count
            for idx, card_count in enumerate(self.table.vec[player_id, :].to(int).cpu().numpy().tolist()):
                [player_table.append(Card(idx).name) for i in range(card_count)]

            player_hands.append(player_hand)
            player_tables.append(player_table)

        # check if names were passed
        if l_player_names is not None:

            string = ''.join([("\033[93m" + str(l_player_names[i]) + "\033[00m").ljust(GAME_DISP_COL_WID+10) for i in l_player_idx]) + '\n\n'

        else:

            string = ''.join([("\033[93mPlayer_" + str(i) + "\033[00m").ljust(GAME_DISP_COL_WID+10) for i in l_player_idx]) + '\n\n'

        # print points
        points = self.table.get_confirmed_player_points().cpu().numpy().tolist()
        string += "\033[34mPOINTS\033[00m\n"
        string += ''.join([str(points[player_idx]).ljust(GAME_DISP_COL_WID) for player_idx in l_player_idx]) + '\n'

        # print hands
        if print_hands:
            string += "\n\033[34mHANDS\033[00m\n"
            for card_idx in range(len(player_hands[0])):
                string += ''.join([(player_hands[player_idx][card_idx]).ljust(GAME_DISP_COL_WID) for player_idx in l_player_idx]) + '\n'

        if len(player_tables[0]) != 0:
            string += "\n\033[34mCARDS ON TABLE\033[00m\n"

            # print tables
            for card_idx in range(len(player_tables[0])):
                string += ''.join([(player_tables[player_idx][card_idx]).ljust(GAME_DISP_COL_WID) for player_idx in l_player_idx]) + '\n'

        return string + '\n'
