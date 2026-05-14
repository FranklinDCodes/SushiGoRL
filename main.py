from collections import deque, namedtuple
import random
import numpy as np
from game import *
from agent import *

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


# HYPERPARAMETERS

ROUND_COUNT = 1e6
MAX_MEMORY = 2e4
BATCH_SIZE = 512
TAU = 0.05
GAMMA = 1



# CONSTANTS

POSSIBLE_PLAYER_COUNTS = [2, 3, 4, 5]
SEED = 64



# build env
env = SushiGo(random.choice(POSSIBLE_PLAYER_COUNTS), SEED)


# build agent
replay_buffer = MemoryBuffer()


# build npcs



for ep in range(ROUND_COUNT):

    # get starting env information
    tup_last_game_states = env.get_states()

    # get starting scores
    last_agent_score = env.get_scores()[0]

    # take steps in episode
    episode_over = False
    while not episode_over:

        # get agent action


        # get npc actions

        # take actions

        # Choose 2nd card if chopsticks used
        # will need to create new env func to get states for this
        # one that takes further chopsticks out of possible actions

        # get new resulting states
        tup_game_states = env.get_states()

        # get rewards
        agent_score = env.get_scores()[0]
        reward = last_agent_score - agent_score

        # save history
        replay_buffer.push()

        # optimize

        # update target net

        # reshuffle deck if that was the third round
        if env.round_num == 3:
            env.setup_new_game()

        # else just setup new round
        else:
            env.setup_new_round()



