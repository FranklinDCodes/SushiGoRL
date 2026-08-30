
from global_constants import *
import torch
import pandas as pd



class Metrics:

    def __init__(self):

        self.agent_scores_period = list()
        self.losses_period = list()

        self.agent_scores_history = list()
        self.losses_history = list()

        self.average_npc_score = {
            'sum': 0,
            'count': 0
        }

    def aggregate(self, game_scores: list[torch.Tensor], loss: float) -> None:
        # adds stats to current period lists

        # update base stats
        self.agent_scores_period.append(game_scores[AGENT_TABLE_POS].detach().cpu().item())
        self.average_npc_score['sum'] += game_scores[[i for i in range(game_scores.size(0)) if i != AGENT_TABLE_POS]].sum()
        self.average_npc_score['count'] += game_scores.size(0) - 1
        self.losses_period.append(loss)

    def commit_current_to_history(self) -> None:
        # averages together current peirod lists and adds them as a history point
        # clears period data

        self.agent_scores_history.append(sum(self.agent_scores_period) / len(self.agent_scores_period))
        self.losses_history.append(sum(self.losses_period) / len(self.losses_period))

        self.agent_scores_period = list()
        self.losses_period = list()

    def save_to_txt(self, filename: str) -> None:

        avg_agent_score = self.average_npc_score['sum'] / self.average_npc_score['count']

        periods = list(range(len(self.losses_history)))

        with open(filename, 'w') as fl:
            fl.writelines([f"Average npc score: {avg_agent_score}\n", 
                           "Agent\n", 
                           "Period\tScore\tLoss" + '\n'])
            fl.writelines([str(periods[i]) + '\t' + str(self.agent_scores_history[i]) + '\t' + str(self.losses_history[i]) + '\n' for i in range(len(self.losses_history))])
