
import numpy as np
import torch
import torch.nn as nn
from global_constants import *
from torch.nn.utils.rnn import pad_sequence


class BaseDQN(nn.Module):

    # player state size is size of state representing each opposing player
    # game state size is size of all other information in game, namely the agent's cards themselves
    def __init__(self):

        super().__init__()

    def forward(self, state: any) -> None:

        raise NotImplementedError(f"Forward function for Deep Q Network was not implemented")
        
    def max_q_action(self, state: any) -> int:

        raise NotImplementedError(f"Action selection function for Deep Q Network was not implemented")
    
    def get_optimizer(self, epoch: int) -> torch.optim.Optimizer:

        raise NotImplementedError(f"Optimizer selection function for Deep Q Network was not implemented")

    def _get_input_tensors_from_state(self, state: PlayerState):

        # state is PlayerState object
        # state is expected to be either a single state where hand is (15), table is (p, 15) where p is 2, 3, 4, or 5?
        # OR a list of such states

        # tensorize hand and add batch dim
        t_hand = state.hand

        # pad table if 2D
        if len(state.hand[0].shape) > 0:

            # calc npc counts
            t_npc_counts = torch.tensor([i.shape[0] - 1 for i in state.table], dtype=torch.int64, device=self.device)
            max_player_count = t_npc_counts.max() + 1

            # rectangularize table
            t_table = pad_sequence(state.table, batch_first=True).to(self.device)

            # add batch dim to hand
            t_hand = torch.stack(state.hand, dim=0).to(self.device)

        else:

            # there is only 1 npc count
            t_npc_counts = torch.tensor([state.table.shape[0] - 1], dtype=torch.int64, device=self.device)
            max_player_count = t_npc_counts[0] + 1

            # table tensor with batch dim
            t_table = state.table.unsqueeze(0).to(self.device)

            # add batch dim to hand
            t_hand = state.hand.unsqueeze(0).to(self.device)

        return t_hand, t_table, t_npc_counts, max_player_count

            

        

