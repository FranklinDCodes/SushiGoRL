
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

        self.rolling_loss = {
            10: {
                'history': list(),
                'window': deque(maxlen=10)
            },
            100: {
                'history': list(),
                'window': deque(maxlen=100)
            },
            1000: {
                'history': list(),
                'window': deque(maxlen=1000)
            },
            10000: {
                'history': list(),
                'window': deque(maxlen=1000)
            }
        }

        self.rolling_points = {
            10: {
                'history': list(),
                'window': deque(maxlen=10)
            },
            100: {
                'history': list(),
                'window': deque(maxlen=100)
            },
            1000: {
                'history': list(),
                'window': deque(maxlen=1000)
            },
            10000: {
                'history': list(),
                'window': deque(maxlen=1000)
            }
        }

    def update(self, game_scores: list[torch.Tensor], loss: float) -> None:

        # update base stats
        self.agent_scores.append(game_scores[AGENT_TABLE_POS].detach().cpu().item())
        self.average_npc_score['sum'] += game_scores[[i for i in range(game_scores.size(0)) if i != AGENT_TABLE_POS]].sum()
        self.average_npc_score['count'] += game_scores.size(0) - 1
        self.losses.append(loss)

        if loss != 0.0:
            pass

        # update rolling stats
        self.rolling_loss = self._update_rolling_stat(loss, self.rolling_loss)
        self.rolling_points = self._update_rolling_stat(game_scores[AGENT_TABLE_POS].item(), self.rolling_points)

    def save_to_txt(self, filename: str) -> None:

        avg_agent_score = self.average_npc_score['sum'] / self.average_npc_score['count']

        epochs = list(range(len(self.losses)))

        with open(filename, 'w') as fl:
            fl.writelines([f"Average npc score: {avg_agent_score}\n", 
                           "Agent\n", 
                           "Epoch\tScore\tLoss\t" + "\t".join(["Loss SMA " + str(i) for i in self.rolling_loss.keys()]) + "\t".join(["Points SMA " + str(i) for i in self.rolling_loss.keys()]) + '\n'])
            fl.writelines([str(epochs[i]) + '\t' + str(self.agent_scores[i]) + '\t' + str(self.losses[i]) + '\t' + '\t'.join([str(average['history'][i]) for average in self.rolling_loss.values()]) + '\t' + '\t'.join([str(average['history'][i]) for average in self.rolling_points.values()]) + '\n' for i in range(len(self.losses))])

    def _update_rolling_stat(self, new_item: any, saved_data: dict) -> dict:

        for window_size in saved_data.keys():

            # add new item
            saved_data[window_size]['window'].append(new_item)

            # check if window is full
            if len(saved_data[window_size]['window']) < window_size:

                # add placeholder item
                saved_data[window_size]['history'].append(0)

            else:

                # average
                total = sum(saved_data[window_size]['window'])

                # add new item
                saved_data[window_size]['history'].append(total / window_size)

        return saved_data
