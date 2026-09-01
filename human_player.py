
from global_constants import *



class HumanPlayer:

    def select_action(self, state: PlayerState) -> Action:

        valid_input = False
        while not valid_input:

            # print options
            print("Possible actions")
            for idx, i in enumerate(state.possible_actions):
                if i == 12:
                    print(f"({idx+1}) Use Chopsticks")
                else:
                    print(f"({idx+1}) Take {Card[i]}")
            print()

            # get input
            text_choice = input("Select card: ")

            # convert
            try:
                idx_choice = int(text_choice) - 1
                choice = state.possible_actions[idx_choice]

            except (KeyError, ValueError):
                print(f"Invalid choice. Please try again.\n")

            else:
                valid_input = True

        return choice




