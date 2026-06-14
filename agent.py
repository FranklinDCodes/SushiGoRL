
import torch
import torch.nn as nn
import numpy as np
from random import random
from copy import deepcopy
from global_constants import *


class RLAgent:

    def __init__(self, model: nn.Module, epsilon_function: any):

        # save model instances
        self.policy_net = model
        self.target_net = deepcopy(model)

        # update count
        self.update_count = 0

        # save epsilon function
        # it's assumed that all epsilon functions map update count to epsilon value
        # recall that epsilon of 1 fully exploits and epsilon of 0 fully explores
        self.epsilon_func = epsilon_function


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
        sample = random.random()
        eps_threshold = self.epsilon_func(self.update_count)

        # if exploit
        if sample > eps_threshold:
            with torch.no_grad():
                
                # return chosen action of policy net based on state
                return Action(self.policy_net.max_q_action(state))
            
        else:

            return np.random.choice(state.possible_actions)
