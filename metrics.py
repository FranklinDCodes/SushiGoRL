
from global_constants import *
import torch
import pandas as pd



class Metrics:

    def __init__(self):

        self.agent_scores = list()
        self.losses = list()
        self.average_npc_score = {
            'sum': 0,
            'count': 0
        }

    def update(self, game_scores: list[torch.Tensor], loss: float) -> None:

        self.agent_scores.append(game_scores[AGENT_TABLE_POS].detach().cpu().item())

        self.losses.append(loss)

        self.average_npc_score['sum'] += game_scores[[i for i in range(game_scores.size(0)) if i != AGENT_TABLE_POS]].sum()
        self.average_npc_score['count'] += game_scores.size(0) - 1

    def save_to_txt(self, filename: str) -> None:

        avg_agent_score = self.average_npc_score['sum'] / self.average_npc_score['count']

        epochs = list(range(len(self.losses)))

        with open(filename, 'w') as fl:
            fl.writelines([f"Average npc score: {avg_agent_score}\n", "Agent\n", "Epoch\tScore\tLoss\n"])
            fl.writelines([str(epochs[i]) + '\t' + str(self.agent_scores[i]) + '\t' + str(self.losses[i]) + '\n' for i in range(len(self.losses))])


