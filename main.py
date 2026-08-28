
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
from schedulers import get_scheduler


# much of the code in this repo is modeled after the following tutorial
# some snippets are copied directly and edited, such as the MemoryBuffer class and Timestep object
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html



# CONFIG
config_name = "config_1"
config_path = f"configs/{config_name}.json"

with open(config_path, 'r') as fl:
    CFG = json.load(fl)


# LOGGING AND SAVING
LOG_PATH = f"logs/{config_name}_{datetime.datetime.now().strftime('%m_%d_%Y_%H-%M-%S')}.log"
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
    CFG["EPS_func"]["name"],
    **CFG["EPS_func"]["kwargs"]
)
SEED = 64



# CONSTANTS



def train_model():

    # build env
    random.seed(SEED)
    env = SushiGo(SEED)

    # load model class dynamically
    BASE_MODEL_PATH = "model_classes"
    MODEL_CLASS_NAME = CFG["model"]["class"]
    spec = importlib.util.spec_from_file_location(MODEL_CLASS_NAME + '.py', os.path.join(BASE_MODEL_PATH, MODEL_CLASS_NAME + '.py'))
    ModelModule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ModelModule)
    ModelClass = ModelModule.DQN

    # device
    DEVICE = CFG["device"]

    # init model
    lr_scheduler = get_scheduler(CFG["lr"]["scheduler_function"], **CFG["lr"]["kwargs"])
    model = ModelClass(
        len(Action), 
        len(Card),
        CARD_NUM,
        max(POSSIBLE_PLAYER_COUNTS),
        lr_scheduler,
        device=DEVICE)

    # build agent
    replay_buffer = MemoryBuffer(CFG["max_memory"])
    agent = RLAgent(model, epsilon_func)

    # build npcs
    l_all_npcs = list()
    for npc in CFG["npcs"]:
        new_npc = get_npc(npc["name"], **npc["kwargs"])
        l_all_npcs.append(new_npc)

    for ep in range(ROUND_COUNT):

        # get player count
        player_count = random.choice(POSSIBLE_PLAYER_COUNTS)

        # random select npcs and order
        random.shuffle(l_all_npcs)
        l_game_npcs = l_all_npcs[:player_count - 1]

        # setup game
        # kind of has some unnecessary computation in this function for some of the rounds
        env.setup_new_game(player_count)

        # get starting env information
        tup_game_states = env.get_states()
        agent_game_state = tup_game_states[AGENT_TABLE_POS]
        npc_game_states = [i for idx, i in enumerate(tup_game_states) if idx != AGENT_TABLE_POS]

        # get starting scores
        last_agent_score = 0

        # take steps in episode
        episode_over = False
        losses = []
        while not episode_over:

            # get agent action
            action = agent.select_action(agent_game_state)

            # get npc actions
            l_npc_actions = []
            for i in range(player_count - 1):
                npc_state = npc_game_states[i]
                l_npc_actions.append(l_game_npcs[i].select_action(npc_state))
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
            l_played_chopsticks = np.arange(player_count - 1)[np_npc_used_chopsticks].tolist()
            if np.any(np_npc_used_chopsticks):

                # call function to manage npc chopsticks use
                np_npc_actions = use_chopsticks_npc(
                    np_npc_actions,
                    l_played_chopsticks,
                    npc_game_states,
                    l_game_npcs,
                    env
                )

            # take actions
            np_all_player_cards = np.array([action, *np_npc_actions])
            env.play_cards(np_all_player_cards)

            # pass hands
            env.pass_hands()

            # check if round is over to determine if there is a real resulting state (for the memory)
            if env.round_is_over():

                new_agent_game_state = None

            else:

                # get new resulting states
                tup_new_game_states = env.get_states() 
                new_agent_game_state = tup_new_game_states[0]

            # get rewards
            agent_score = env.get_scores()[0]
            reward = agent_score - last_agent_score

            # save history
            replay_buffer.push(agent_game_state, action, new_agent_game_state, reward)

            # setup new round
            if env.round_is_over() and env.round_num != 3:
                
                env.setup_new_round(player_count)

                # get game states
                tup_new_game_states = env.get_states()
                new_agent_game_state = tup_new_game_states[0]
                agent_score = 0

            # end game
            elif env.round_is_over() and env.round_num == 3:
                episode_over = True

            # set old states
            tup_game_states = tup_new_game_states
            agent_game_state = new_agent_game_state
            last_agent_score = agent_score
            npc_game_states = [i for idx, i in enumerate(tup_game_states) if idx != AGENT_TABLE_POS]

            # optimize
            loss = optimize(BATCH_SIZE, replay_buffer, agent.policy_net, agent.target_net, GAMMA, ep, DEVICE)
            if loss is not None:
                losses.append(loss)

            # update target net
            agent.soft_update_target_net(TAU)

        if (ep+1) % 10 == 0:
            print(ep+1)


if __name__ == "__main__":
    train_model()

