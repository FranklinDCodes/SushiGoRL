
from base_model import BaseDQN
import torch
import torch.nn as nn
import numpy as np


class DQN(BaseDQN):

    # player state size is size of state representing each opposing player
    # game state size is size of all other information in game, namely the agent's cards themselves
    def __init__(self, 
                n_actions: int, 
                n_table_card_types: int,
                n_hand_card_types: int,
                max_players: int,
                lr_scheduler: any = lambda ep: 1e-3,

                embedded_player_position_size: int = 16,
                encoded_player_cards_size: int = 128,
                
                relu_leak: float = 1e-2
                ):

        super().__init__()

        # positional embedding
        # player_num -> pos_embedding
        self.pos_embed = nn.Embedding(max_players, embedded_player_position_size)

        # hand encoder
        # player_positional_card_counts X pos_embedding -> encoded_player_cards
        self.player_cards_encoder = nn.Sequential(
            nn.Linear(n_table_card_types + embedded_player_position_size, 64),
            nn.LeakyReLU(relu_leak),
            nn.Linear(64, 128),
            nn.LeakyReLU(relu_leak),
            nn.Linear(128, encoded_player_cards_size)
        )

        # return approximation network
        # pooled_npc_encoded_cards X encoded_agent_cards X agent_hand -> card_return
        self.return_estimator = nn.Sequential(
            nn.Linear(encoded_player_cards_size * 2 + n_hand_card_types, 128),
            nn.LeakyReLU(relu_leak),
            nn.Linear(128, 64),
            nn.LeakyReLU(relu_leak),
            nn.Linear(64, 16),
            nn.LeakyReLU(relu_leak),
            nn.Linear(16, n_actions)
        )

        # training objects
        self.loss_function = nn.MSELoss()
        self.lr_scheduler = lr_scheduler
        self.optim_class = torch.optim.Adam

    def forward(self, state: any) -> np.ndarray:
        
        # get positional encoding
        player_count = state.table.shape[-2]
        t_pos_embeddings = self.pos_embed.forward(torch.arange(player_count))

        # encode player played cards
        t_player_card_states = torch.concat((torch.tensor(state.table), t_pos_embeddings), axis=1)
        t_player_cards_encoded = self.player_cards_encoder.forward(t_player_card_states)

        # pool player cards 
        t_agent_cards_encoded = t_player_cards_encoded[..., 0, :]
        t_npc_cards_encoded = t_player_cards_encoded[..., 1:, :]
        t_npc_cards_pooled = t_npc_cards_encoded.sum(axis=0)

        # Run estimator
        t_hand = torch.tensor(state.hand)
        t_estimator_input = torch.concat((t_hand, t_agent_cards_encoded, t_npc_cards_pooled), axis=0).to(torch.float32)
        np_pred_returns = self.return_estimator.forward(t_estimator_input).numpy()

        return np_pred_returns
        
    def max_q_action(self, state: any) -> int:

        # batch dim

        np_pred_returns = self.forward(state)


        # FIX THIS TOP ACTION ALLOWED TO BE MASK

        # pick top rated action
        allowed = state.possible_actions
        top_action = np.arange(np_pred_returns.shape[0])[np.isin(np.arange(np_pred_returns.shape[0]), allowed)][np.argmax(np_pred_returns[np.isin(np.arange(np_pred_returns.shape[0]), allowed)])]

        return top_action
    
    def get_optimizer(self, epoch: int) -> torch.optim.Optimizer:

        return self.optim_class(self.parameters(), self.lr_scheduler(epoch))

