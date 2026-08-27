
import torch
import torch.nn as nn
from ..global_constants import *
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

        # hand tensor
        t_hand = torch.tensor(state.hand, dtype=self.dtype, device=self.device)

        # pad table if 2D
        if len(state.hand[0].shape) > 0:

            t_table_sizes = torch.tensor([i.shape[0] for i in state.table], dtype=torch.int64, device=self.device)

            list_t_table = [torch.tensor(i, dtype=self.dtype, device=self.device) for i in state.table]
            t_table = pad_sequence(list_t_table, batch_first=True)

            max_player_count = t_table_sizes.max()

        else:

            t_table_sizes = torch.tensor([state.table.shape[0]], dtype=torch.int64, device=self.device)

            t_table = torch.tensor(state.table, dtype=self.dtype, device=self.device).unsqueeze(0)

            max_player_count = t_table_sizes[0]

        return t_table, t_table_sizes, max_player_count

            

        

