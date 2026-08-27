
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from global_constants import *


def get_next_q_values(
        t_all_next_q: torch.tensor, # timesteps X actions
        lnp_possible_actions: list[np.array], # timesteps X possible_actions(heterogenous)
        t_is_terminal_state_mask: torch.tensor # timesteps
        ):

    l_all_action_masks = []
    for i in range(t_all_next_q.shape[0]):

        # init mask
        t_mask_i = torch.zeros((t_all_next_q.shape[1]), dtype=bool)

        # capture indices of allowed actions
        np_possible_actions = lnp_possible_actions[i]

        # set mask
        t_mask_i[np_possible_actions] = 1
        l_all_action_masks.append(t_mask_i)

    # create 2d mask
    t_all_action_masks = torch.stack(l_all_action_masks)

    # zero out all restricted actions with mask and find max
    t_all_next_q[~t_all_action_masks] = float("-inf")
    t_max_next_qs_non_zero = t_all_next_q.max(dim=1)

    # create new tensor to add in the zeros for the terminal states
    t_all_timestep_max_q = torch.zero_like(t_is_terminal_state_mask, dtype=torch.float32)
    t_all_timestep_max_q[~t_is_terminal_state_mask] = t_max_next_qs_non_zero

    return t_all_timestep_max_q


def optimize(batch_size: int, buffer: MemoryBuffer, policy_net: any, target_net: any, gamma: int, batch_num: int):

    # check if replay buffer is full
    if len(buffer) != buffer.capacity:
        return

    print("Optimizing")

    # sample batch
    timestep_training_data = buffer.sample(batch_size)

    # make tensors of data
    int_action_batch = list(timestep_training_data.action)
    t_action_batch = torch.tensor(int_action_batch, dtype=torch.int16)
    t_reward_batch = torch.tensor(timestep_training_data.reward, dtype=torch.float32)

    # unpack state
    # transpose from a list of states to a state of lists
    state_batch = PlayerState(*zip(*timestep_training_data.state))
    non_final_next_states = [i for i in timestep_training_data.next_state if i is not None]
    next_state_batch = PlayerState(*zip(*non_final_next_states))

    # create mask for final states
    t_final_state_mask = torch.tensor(list(map(lambda s: s is None, timestep_training_data.next_state)))

    # get next state qs
    t_all_next_q = target_net.forward(next_state_batch)

    # get possible action mask for next state
    t_all_max_next_q = get_next_q_values(t_all_next_q, state_batch.possible_actions, t_final_state_mask)

    # create labels
    t_labels = t_reward_batch + gamma * (t_all_max_next_q)
    
    # train
    optimizer = policy_net.get_optimizer(batch_num)
    optimizer.zero_grad()
    t_predictions = policy_net.forward(state_batch) # Batch X Actions
    t_predictions_for_actions_taken = t_predictions[t_action_batch] # Batch
    loss = policy_net.loss_function(t_labels, t_predictions_for_actions_taken)
    loss.backward()
    optimizer.step()

    return loss.item()
