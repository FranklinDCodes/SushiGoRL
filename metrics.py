
from global_constants import *
import torch
import pandas as pd



class Metrics:

    def __init__(self):

        self.d_period_data = {
            'scores': list(),
            'losses': list()
        }

        self.d_metric_history = {
            'scores': list(),
            'losses': list()
        }

        self.average_npc_score = {
            'sum': 0,
            'count': 0
        }

    def aggregate(self, key: str, item: any) -> None:
        # adds stats to current period lists

        if key == 'npc_score':
            # if updating npc average score
            # this is universally tracked so just add all the items from the tensor and add the count

            self.average_npc_score['sum'] += item[[i for i in range(item.size(0)) if i != AGENT_TABLE_POS]].sum()
            self.average_npc_score['count'] += item.size(0) - 1

        else:

            self.d_period_data[key].append(item)

    def commit_current_to_history(self) -> None:
        # averages together current peirod lists and adds them as a history point
        # clears period data

        for key, l_values in self.d_period_data.items():

            # add average
            avg = sum(l_values) / len(l_values)
            self.d_metric_history[key].append(avg)

            # clear data from collection period
            self.d_period_data[key] = list()

    def save_to_txt(self, filename: str) -> None:

        avg_agent_score = self.average_npc_score['sum'] / self.average_npc_score['count']

        periods = list(range(len(self.d_metric_history['losses'])))

        with open(filename, 'w') as fl:
            fl.writelines([f"Average npc score: {avg_agent_score}\n", 
                           "Agent\n", 
                           "Period\tLoss\tScore" + '\n'])
            fl.writelines([str(periods[i]) + '\t' + str(self.d_metric_history['losses'][i]) + '\t' + str(self.d_metric_history['scores'][i]) + '\n' for i in periods])
