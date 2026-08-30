
import torch.nn as nn
import torch.optim as optim


class ConstantScheduler:

    def __init__(self, rate: float):

        self.rate = rate

    def __call__(self, epoch: int):

        return self.rate
    

schedulers_dict = {
    "constant": ConstantScheduler
}

def get_scheduler(cfg: dict):

    if cfg is None:
        return ConstantScheduler()

    return schedulers_dict.get(cfg.get("scheduler_function"), ConstantScheduler)(**cfg.get("kwargs", {}))


optimizer_dict = {
    "adam": optim.Adam,
    "adamw": optim.AdamW
}

def get_optimizer_class(optim_name: str):

    return optimizer_dict.get(optim_name, optim.Adam)


loss_dict = {
    "mse": nn.MSELoss,
    "huber": nn.SmoothL1Loss
}

def get_loss(cfg: dict):

    if cfg is None:
        return nn.MSELoss()
    
    return loss_dict.get(cfg.get("class"), nn.MSELoss)(**cfg.get("kwargs", {}))

