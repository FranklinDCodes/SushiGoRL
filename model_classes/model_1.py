
from model_classes.base_model import BaseDQN
import torch
import torch.nn as nn


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
                
                relu_leak: float = 1e-2,

                dtype: any = torch.float32,
                device: str = 'cpu'
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

        # settings
        self.dtype = dtype
        self.device = device

        self.to(self.device)

    def forward(self, state: any) -> torch.Tensor:

        t_hand, t_table, t_npc_counts, max_player_count = self._get_input_tensors_from_state(state)
        
        # get positional encoding
        t_player_range = torch.arange(max_player_count, dtype=torch.int64, device=self.device)
        t_pos_embeddings = self.pos_embed.forward(t_player_range)

        # add batch dim to positional embeddings
        batch_size = t_table.size(0)
        t_pos_embeddings = t_pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # encode player played cards
        t_player_card_states = torch.concat((t_table, t_pos_embeddings), axis=-1)
        t_player_cards_encoded = self.player_cards_encoder.forward(t_player_card_states)

        # separate encodings into PC and NPC
        t_agent_cards_encoded = t_player_cards_encoded[..., 0, :]
        t_npc_cards_encoded = t_player_cards_encoded[..., 1:, :]

        # create mask of valid npc states
        max_npc_count = max_player_count - 1
        t_real_npc_mask = torch.arange(max_npc_count).to(self.device) < t_npc_counts.unsqueeze(-1)
        t_real_npc_mask = t_real_npc_mask.unsqueeze(-1)

        # 0 out npc states that are invalid and pool
        t_npc_cards_encoded *= t_real_npc_mask
        t_npc_cards_pooled = t_npc_cards_encoded.sum(axis=-2)

        # Run estimator
        t_estimator_input = torch.concat((t_hand, t_agent_cards_encoded, t_npc_cards_pooled), axis=-1)
        t_pred_returns = self.return_estimator.forward(t_estimator_input)

        return t_pred_returns
        
    def max_q_action(self, state: any) -> int:

        # forward
        t_pred_returns = self.forward(state)

        # eliminate batch dim
        t_pred_returns = t_pred_returns.squeeze()

        # sort best actions
        t_idx_sorted = torch.argsort(t_pred_returns, descending=True)

        # eliminate impossible actions
        t_acceptable_idx_sorted = t_idx_sorted[torch.isin(t_idx_sorted, state.possible_actions)]

        # grab top action
        top_action_int = t_acceptable_idx_sorted[0].detach().cpu().item()

        return top_action_int
    
    def get_optimizer(self, epoch: int) -> torch.optim.Optimizer:

        return self.optim_class(self.parameters(), lr=self.lr_scheduler(epoch))

