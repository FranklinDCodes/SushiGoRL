
import torch
import torch.nn as nn
from random import random
from copy import deepcopy
from global_constants import *


class RLAgent:

    def __init__(self, model: nn.Module, epsilon_function: any, seed: int):

        # save model instances
        self.policy_net = model
        self.target_net = deepcopy(model)

        # update count
        self.update_count = 0

        # save epsilon function
        # it's assumed that all epsilon functions map update count to epsilon value
        # recall that epsilon of 1 fully exploits and epsilon of 0 fully explores
        self.epsilon_func = epsilon_function

        self.rand_gen = torch.Generator()
        self.rand_gen = self.rand_gen.manual_seed(seed)
        
    def soft_update_target_net(self, TAU: float):

        # TAU here is percent of the update to do
        # e.g. TAU = 1 is full update, TAU = 0 is non-update

        # grab state dicts
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()

        # do a weighted sum of both nets for each param in target net
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        
        # set target net
        self.target_net.load_state_dict(target_net_state_dict)

    def select_action(self, state: PlayerState) -> Action:

        # check epsilon value
        sample = torch.rand(1, generator=self.rand_gen).item()
        eps_threshold = self.epsilon_func(self.update_count)
        self.update_count += 1

        # if exploit
        if sample > eps_threshold:
            with torch.no_grad():
                
                # return chosen action of policy net based on state
                return self.policy_net.max_q_action(state)
            
        else:

            # generate random scalar index with generator
            rand_idx = torch.randint(state.possible_actions.size(0), (1, ), generator=self.rand_gen).item()

            # return item
            return state.possible_actions[rand_idx]
