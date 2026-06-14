
# libraries
from collections import deque, namedtuple
import random
import numpy as np
import json
import os
import datetime
import importlib

# source code
from global_constants import *
from game import *
from epsilon import *
from agent import *
from npc import *
from chopsticks import *
from optimization import *


# much of the code in this repo is modeled after the following tutorial
# some snippets are copied directly and edited, such as the MemoryBuffer class and Timestep object
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html



# CONFIG
config_name = "train_config_1"
config_path = f"configs/{config_name}"

with open(config_path, 'r') as fl:
    CFG = json.load(fl)


# LOGGING AND SAVING
LOG_PATH = f"logs/{config_name}_{datetime.datetime.now().strftime('%m_%d_%Y_%H-%M-$S%')}.log"
def log(s: any) -> None:

    print(s)
    with open(LOG_PATH, 'a') as fl:
        fl.write(str(s))



# HYPERPARAMETERS

ROUND_COUNT = CFG["round_count"]
MAX_MEMORY = CFG["max_memory"]
BATCH_SIZE = CFG["batch_size"]
TAU = CFG["TAU"]
GAMMA = CFG["GAMMA"]
epsilon_func = get_epsilon_function(
    CFG["EPS_func"],
    **CFG["kwargs"]
)
SEED = 64



# CONSTANTS



def train_model():

    # build env
    env = SushiGo(random.choice(POSSIBLE_PLAYER_COUNTS), SEED)

    # load model class dynamically
    BASE_MODEL_PATH = "model_classes"
    MODEL_CLASS_NAME = CFG["model"]["class"]
    spec = importlib.util.spec_from_file_location(MODEL_CLASS_NAME + '.py', os.path.join(BASE_MODEL_PATH, MODEL_CLASS_NAME + '.py'))
    ModelModule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ModelModule)
    ModelClass = ModelModule.DQN

    # init model
    model = ModelClass(
        len(Action), 
        len(Card),
        CARD_NUM,
        max(POSSIBLE_PLAYER_COUNTS))

    # build agent
    replay_buffer = MemoryBuffer()
    agent = RLAgent(model, epsilon_func)

    # build npcs
    l_npcs = list()
    for npc in CFG["npcs"]:
        l_npcs.append(get_npc(npc["name"], **npc["kwargs"]))

    for ep in range(ROUND_COUNT):

        # get player count
        player_count = random.choice(POSSIBLE_PLAYER_COUNTS)

        # random sample indexes for npcs
        npc_table_seat_nums = random.sample(list(range(player_count - 1)), k=player_count)

        # get starting env information
        tup_game_states = env.get_states()
        agent_game_state = tup_game_states[AGENT_TABLE_POS]

        # get starting scores
        last_agent_score = env.get_scores()[AGENT_TABLE_POS]

        # take steps in episode
        episode_over = False
        while not episode_over:

            # get agent action
            action = agent.select_action(agent_game_state)

            # get npc actions
            l_npc_actions = []
            for idx, npc_num in enumerate(npc_table_seat_nums):
                l_npc_actions.append(l_npcs[npc_num].select_action(tup_game_states[idx]).action)
            np_npc_actions = np.array(l_npc_actions)

            # check if agent used chopsticks
            chopsticks_played = action == Action.PlayChopsticks.value
            if chopsticks_played:
                
                # call chopsticks function
                action, agent_game_state, last_agent_score = use_chopsticks_agent(
                    agent_game_state,
                    agent,
                    replay_buffer,
                    env
                )

            # check for npc chopsticks
            np_npc_used_chopsticks = np_npc_actions == Action.PlayChopsticks.value
            l_played_chopsticks = np.arange(len(npc_table_seat_nums))[np_npc_used_chopsticks].tolist()
            if np.any(np_npc_used_chopsticks):

                # call function to manage npc chopsticks use
                np_npc_actions = use_chopsticks_npc(
                    npc_table_seat_nums,
                    l_played_chopsticks,
                    tup_game_states,
                    l_npcs,
                    env
                )

            # take actions
            env.play_cards([action, *np_npc_actions.tolist()])

            # check if round is over
            if env.round_is_over():

                new_agent_game_state = None

            else:

                # get new resulting states
                tup_new_game_states = env.get_states()
                new_agent_game_state = tup_new_game_states[0]

            # get rewards
            agent_score = env.get_scores()[0]
            reward = last_agent_score - agent_score

            # save history
            replay_buffer.push(Timestep(agent_game_state, action, new_agent_game_state, reward))

            # set old states
            tup_game_states = tup_new_game_states
            agent_game_state = new_agent_game_state
            last_agent_score = agent_score

            # optimize




            # update target net
            agent.soft_update_target_net(TAU)

        # reshuffle deck if that was the third round
        if env.round_num == 3:
            env.setup_new_game()

        # else just setup new round
        else:
            env.setup_new_round()



if __name__ == "__main__":
    train_model()

