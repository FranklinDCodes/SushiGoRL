
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from shared_objects import *

def train(batch_size: int, buffer: MemoryBuffer, policy_net: any, target_net: any, gamma: int):

    # sample batch
    ts_training_data = buffer.sample(batch_size)

    # create mask for final states
    t_final_state_mask = torch.tensor(list(map(lambda s: s is not None, ts_training_data.next_state)))
    t_non_final_next_states = torch.cat([i for i in ts_training_data.next_state if i is not None])

    # make tensors of data
    t_state_batch = torch.cat(ts_training_data.state)
    t_action_batch = torch.cat(ts_training_data.action)
    t_reward_batch = torch.cat(ts_training_data.reward)

    # forward pass
    t_all_q_hat = policy_net.forward(t_state_batch)

    # get pred qs for only actions chosen
    t_actions_q_hat = t_all_q_hat[torch.arange(t_all_q_hat.shape[0]), t_action_batch]

    # get next state qs
    t_all_next_q = target_net.forward(t_non_final_next_states)
    t_max_next_q = torch.argmax()

    # create labels

    t_labels = t_reward_batch + gamma * ()
    

