
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from global_constants import *

def optimize(batch_size: int, buffer: MemoryBuffer, policy_net: any, target_net: any, gamma: int):

    # sample batch
    timestep_training_data = buffer.sample(batch_size)

    # create mask for final states
    t_final_state_mask = torch.tensor(list(map(lambda s: s is not None, timestep_training_data.next_state)))
    l_non_final_next_states = [i for i in timestep_training_data.next_state if i is not None]

    # make tensors of data
    t_action_batch = torch.cat(timestep_training_data.action)
    t_reward_batch = torch.cat(timestep_training_data.reward)

    # unpack state
    # transpose from a list of states to a state of lists
    state_batch = PlayerState(*zip(*timestep_training_data.state))
    next_state_batch = PlayerState(*zip(*timestep_training_data.next_state))



    # convert state elements to tensors
    state_batch.hand = torch.tensor(state_batch.hand)
    state_batch.table = torch.tensor(state_batch.table)
    state_batch.possible_actions



    # forward pass
    t_all_q_hat = policy_net.forward(state_batch)

    # get pred qs for only actions chosen
    t_actions_q_hat = t_all_q_hat[torch.arange(t_all_q_hat.shape[0]), t_action_batch]

    # get next state qs
    t_all_next_q = target_net.forward(next_state_batch)

    # get possible action mask for next state
    t_max_next_q = torch.max(t_all_next_q[torch.isin(torch.arange(t_all_next_q.shape[0])])

    # create labels

    t_labels = t_reward_batch + gamma * ()
    

