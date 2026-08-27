import torch
import torch.nn as nn



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

    def _get_input_tensors_from_state(self, state: PlayerState)
