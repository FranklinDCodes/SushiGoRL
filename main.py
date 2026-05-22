from collections import deque, namedtuple
import random
import numpy as np
import json
import os
import datetime
import importlib
from game import *
from epsilon import *
from agent import *
from npc import *

Timestep = namedtuple('Timestep',
                        ('state', 'action', 'next_state', 'reward'))

class MemoryBuffer:

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Timestep(*args))

    def sample(self, batch_size: int) -> Timestep:
        samples = random.sample(self.memory, batch_size)
        batched = Timestep(*zip(*samples))
        return batched

    def __len__(self) -> int:
        return len(self.memory)


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



# CONSTANTS

POSSIBLE_PLAYER_COUNTS = [2, 3, 4, 5]
SEED = 64



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
    npc_nums = random.sample(list(range(player_count - 1)), k=player_count)

    # get starting env information
    tup_game_states = env.get_states()
    agent_game_state = tup_game_states[0]

    # get starting scores
    last_agent_score = env.get_scores()[0]

    # take steps in episode
    episode_over = False
    while not episode_over:

        # get agent action
        action = agent.select_action(agent_game_state)

        # get npc actions
        l_npc_actions = []
        for idx, npc_num in enumerate(npc_nums):
            l_npc_actions.append(l_npcs[npc_num].forward(tup_game_states[idx]))
        np_npc_actions = np.array(l_npc_actions)

        # check if agent used chopsticks
        chopsticks_played = action == Action.PlayChopsticks.value
        if chopsticks_played:

            # make copy of the state that is before the first chopstick choice
            state_pre_first_cs_choice = np.copy(agent_game_state)
            state_pre_first_cs_choice.table[Card.Chopsticks.value] -= 1

            # remove chopsticks from possible actions
            state_pre_first_cs_choice.possible_actions = [i for i in state_pre_first_cs_choice.possible_actions if i != Action.PlayChopsticks.value]
    
            # play chopsticks
            replay_buffer.push(Timestep(agent_game_state, action, state_pre_first_cs_choice, 0.0))
            
            # get first chopstick choice
            first_cs_choice = agent.select_action(state_pre_first_cs_choice)

            # play first choice
            env.play_chopsticks(0, first_cs_choice)

            # get new state and remove chopsticks from possible actions
            state_pre_second_cs_choice = env.get_states()[0]
            state_pre_second_cs_choice.possible_actions = [i for i in state_pre_second_cs_choice.possible_actions if i != Action.PlayChopsticks.value]

            # add first chopstick pick to history
            first_choice_reward = env.get_scores()[0] - agent_last_score
            replay_buffer.push(Timestep(state_pre_first_cs_choice, first_cs_choice, state_pre_second_cs_choice, first_choice_reward))

            # reset last score
            agent_last_score = env.get_scores()[0]

            # get 2nd chopstick choice
            action = agent.select_action(state_pre_second_cs_choice)

        # check for npc chopsticks
        if np.any(np_npc_actions == Action.PlayChopsticks.value):

            # get npc actions
            l_npc_actions = []
            for idx, npc_num in enumerate(npc_nums):
                l_npc_actions.append(l_npcs[npc_num].forward(tup_game_states[idx]))
            np_npc_actions = np.array(l_npc_actions)



        # take actions
        env.play_cards([action, *np_npc_actions.tolist()])

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
        agent_last_score = agent_score


        # optimize

        # update target net
        agent.soft_update_target_net(TAU)

        # reshuffle deck if that was the third round
        if env.round_num == 3:
            env.setup_new_game()

        # else just setup new round
        else:
            env.setup_new_round()



