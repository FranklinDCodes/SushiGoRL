
from global_constants import *


PRINT_COLS = 3
ACTION_COL_WID = 25

class HumanPlayer:

    def select_action(self, state: PlayerState) -> Action:

        valid_input = False
        while not valid_input:

            # print options
            print("Possible actions")
            for idx, i in enumerate(state.possible_actions):
                if i == 12:

                    # print use chopsticks
                    print(f"({idx+1}) Use Chopsticks".ljust(ACTION_COL_WID), end=' ')

                else:

                    # print take card action
                    card_count = int(state.hand[i])
                    if card_count > 1:
                        print(f"({idx+1}) Take {Card(i.item()).name} x{card_count}".ljust(ACTION_COL_WID), end=' ')
                    else:
                        print(f"({idx+1}) Take {Card(i.item()).name}".ljust(ACTION_COL_WID), end=' ')

                if (idx+1) % 3 == 0:
                    print()
            print()

            # get input
            text_choice = input("Select card: ")

            # convert
            try:
                idx_choice = int(text_choice) - 1
                choice = state.possible_actions[idx_choice]

            except (KeyError, IndexError, ValueError):
                print(f"Invalid choice. Please try again.\n")

            else:
                valid_input = True

        return choice
