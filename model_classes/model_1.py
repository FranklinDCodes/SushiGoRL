import torch
import torch.nn as nn



class DQN(nn.module):

    # player state size is size of state representing each opposing player
    # game state size is size of all other information in game, namely the agent's cards themselves
    def __init__(self, 
                n_actions: int, 
                n_card_types: int,
                max_players: int,

                embedded_card_size: int = 16,
                encoded_card_size: int = 32,
                embedded_player_position_size: int = 16,
                encoded_player_hand_size: int = 128,
                
                relu_leak: float = 1e-2
                ):

        super(DQN, self).__init__()

        # card embedding
        # card_num -> card_embedding
        card_embed = nn.Embedding(n_card_types, embedded_card_size)

        # card encoder
        # card_embedding -> encoded_card
        card_encoder = nn.Sequential(
            nn.Linear(embedded_card_size, 16),
            nn.LeakyReLU(relu_leak),
            nn.Linear(16, 32),
            nn.LeakyReLU(relu_leak),
            nn.Linear(32, encoded_card_size)
        )

        # positional embedding
        # player_num -> pos_embedding
        pos_embed = nn.Embedding(max_players, embedded_player_position_size)

        # hand encoder
        # pooled_player_encoded_cards X pos_embedding -> encoded_player_hand
        player_hand_encoder = nn.Sequential(
            nn.Linear(encoded_card_size + embedded_player_position_size, 64),
            nn.LeakyReLU(relu_leak),
            nn.Linear(64, 128),
            nn.LeakyReLU(relu_leak),
            nn.Linear(128, encoded_player_hand_size)
        )

        # return approximation network
        # pooled_npc_encoded_hands X encoded_agent_hand X card_under_consideration -> card_return
        return_estimator = nn.Sequential(
            nn.Linear(encoded_player_hand_size * 2 + encoded_card_size, 128),
            nn.LeakyReLU(relu_leak),
            nn.Linear(128, 32),
            nn.LeakyReLU(relu_leak),
            nn.Linear(32, 1)
        )

    def forward():
        pass