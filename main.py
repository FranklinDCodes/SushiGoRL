from collections import deque, namedtuple
import random
import numpy as np
from game import *




# Timestep and MemoryBuffer objects mostly stolen from
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

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



# HYPERPARAMETERS

ROUND_COUNT = 1e6
MAX_MEMORY = 2e4
BATCH_SIZE = 512 


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



