
# FINISH


import torch
import json
import random
import importlib
import os
import sys
import time

from agent import *
from global_constants import *
from game import *
from npc import *
from chopsticks import *
from human_player import HumanPlayer

CONFIG_PATH = "configs/inference_configs/config_model_12.json" # sys.argv[1]
DEVICE = 'cpu'
PAUSE = False


def main(): 

    with open(CONFIG_PATH, 'r') as fl:
        CFG = json.load(fl)

    SAVE_PATH = CFG["model"]["save_path"]
    PLAYER_COUNT = CFG["player_count"]

    SEED = CFG["seed"]
    DEVICE = "cpu"

    # load model class dynamically
    BASE_MODEL_PATH = "model_classes"
    MODEL_CLASS_NAME = CFG["model"]["class"]
    spec = importlib.util.spec_from_file_location(MODEL_CLASS_NAME + '.py', os.path.join(BASE_MODEL_PATH, MODEL_CLASS_NAME + '.py'))
    ModelModule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ModelModule)
    ModelClass = ModelModule.DQN

    # seed
    random.seed(SEED)
    torch.manual_seed(SEED+1)

    # init env
    env = SushiGo(SEED+2, DEVICE)

    # init model
    model = ModelClass(
        len(Action), 
        len(Card),
        CARD_NUM,
        max(POSSIBLE_PLAYER_COUNTS),
        None,
        None,
        None,
        device=DEVICE)

    # load model
    state_dict = torch.load(SAVE_PATH)
    model.load_state_dict(state_dict)

    # build npcs
    l_all_npcs = list()
    l_npc_configs = CFG["npcs"]
    for npc_cfg in l_npc_configs:
        new_npc = get_npc(npc_cfg["name"], **npc_cfg["kwargs"])
        l_all_npcs.append(new_npc)

    # name each player
    ai_agent_name = "SushiGoatAI"
    l_npc_names = [npc_cfg['name'] + "_npc" for i in l_npc_configs]
    human_player_name = "You"

    # build agent
    epsilon_func = lambda x: 1
    agent = RLAgent(model, epsilon_func, SEED+3)

    # init human player
    human_player = HumanPlayer()

    while True:

        # select npcs that are playing
        l_npcs_playing_idx = random.sample(range(len(l_all_npcs)), PLAYER_COUNT - 2)
        l_game_npcs = [i for idx, i in enumerate(l_all_npcs) if idx in l_npcs_playing_idx]
        l_game_npc_names = [i for idx, i in enumerate(l_npc_names) if idx in l_npcs_playing_idx]

        # create list with npcs and human player and shuffle
        l_non_ai_players_unshuffled = [human_player, *l_game_npcs]
        l_non_ai_player_names_unshuffled = [human_player_name, *l_game_npc_names]
        l_non_at_player_order = random.sample(range(PLAYER_COUNT - 1), PLAYER_COUNT - 1)
        l_game_non_ai_players = [l_non_ai_players_unshuffled[i] for i in l_non_at_player_order]
        l_game_non_ai_names = [l_non_ai_player_names_unshuffled[i] for i in l_non_at_player_order]
        human_idx = l_non_at_player_order.index(0) + 1      # add 1 for ai agent

        # setup game
        env.setup_new_game(PLAYER_COUNT)

        # get starting env information
        tup_game_states = env.get_states()
        agent_game_state = tup_game_states[AGENT_TABLE_POS]
        non_ai_game_states = [i for idx, i in enumerate(tup_game_states) if idx != AGENT_TABLE_POS]

        # take steps in episode
        episode_over = False
        while not episode_over:

            print()
            print()
            print()
            print(env.__str__(False, human_idx, [ai_agent_name, *l_game_non_ai_names]))
            if PAUSE:
                time.sleep(1.5)

            # get agent action
            action = agent.select_action(agent_game_state)

            # get non-ai actions
            l_non_ai_actions = []
            for i in range(PLAYER_COUNT - 1):
                state = non_ai_game_states[i]
                l_non_ai_actions.append(l_game_non_ai_players[i].select_action(state))
            t_non_ai_actions = torch.tensor(l_non_ai_actions, device=DEVICE)

            # check if agent used chopsticks
            chopsticks_played = action == Action.PlayChopsticks.value
            if chopsticks_played:
                
                # call chopsticks function
                action, agent_game_state, last_agent_score = use_chopsticks_agent(
                    agent_game_state,
                    agent,
                    None,
                    env
                )

            # check for non-agent chopsticks
            t_non_ai_used_chopsticks = t_non_ai_actions == Action.PlayChopsticks.value
            l_played_chopsticks = torch.arange(PLAYER_COUNT - 1)[t_non_ai_used_chopsticks.cpu()].tolist()
            if torch.any(t_non_ai_used_chopsticks):

                # call function to manage npc chopsticks use
                t_non_ai_actions = use_chopsticks_npc(
                    t_non_ai_actions,
                    l_played_chopsticks,
                    non_ai_game_states,
                    l_game_non_ai_players,
                    env
                )

            # take actions
            t_all_player_cards = torch.tensor([action, *t_non_ai_actions], device=DEVICE)
            env.play_cards(t_all_player_cards)

            # pass hands
            env.pass_hands()

            # check if round is over to determine if there is a real resulting state (for the memory)
            if env.round_is_over():

                new_agent_game_state = None

            else:

                # get new resulting states
                tup_new_game_states = env.get_states() 
                new_agent_game_state = tup_new_game_states[0]

            if PAUSE:
                time.sleep(1.5)

            # setup new round
            if env.round_is_over() and env.round_num != 3:

                print()
                print()
                print()
                print("\033[93mROUND OVER\033[00m")
                print()
                print(''.join([("\033[93m" + str(l_player_names[i]) + "\033[00m").ljust(30+10) for i in l_player_idx]))
                print("\033[34mROUND FINAL SCORE\033[00m")
                l_player_idx = [(i + human_idx) % PLAYER_COUNT for i in range(PLAYER_COUNT)]
                l_player_names = [ai_agent_name, *l_game_non_ai_names]
                points = env.get_round_scores()
                print(''.join([str(points[player_idx].item()).ljust(30) for player_idx in l_player_idx]))
                print()
                
                env.setup_new_round(PLAYER_COUNT)

                print("\033[34mGAME SCORE\033[00m")
                points = env.get_game_scores()
                print(''.join([str(points[player_idx].item()).ljust(30) for player_idx in l_player_idx]))
                print()
                print()

                # get game states
                tup_new_game_states = env.get_states()
                new_agent_game_state = tup_new_game_states[0]

            # end game
            elif env.round_is_over() and env.round_num == 3:
                episode_over = True
                env.end_game()

                print()
                print("\033[93mGAME OVER\033[00m")
                print()
                print("\033[34mFINAL SCORE\033[00m")
                l_player_idx = [(i + human_idx) % PLAYER_COUNT for i in range(PLAYER_COUNT)]
                points = env.get_game_scores()
                print(''.join([str(points[player_idx]).ljust(30) for player_idx in l_player_idx]))
                print()

            # set old states
            tup_game_states = tup_new_game_states
            agent_game_state = new_agent_game_state
            non_ai_game_states = [i for idx, i in enumerate(tup_game_states) if idx != AGENT_TABLE_POS]

        print(f"GAME OVER")




if __name__ == "__main__":
    main()

