
# FINISH


import torch
import json
import random
import importlib

from global_constants import *
from game import *
from npc import *

CONFIG_PATH = "configs/configs_2.json"
SAVE_PATH = "outcomes/config_2_08_30_2026_02_03_02/model_save.pkl"


def main(): 

    with open(CONFIG_PATH, 'r') as fl:
        CFG = json.load(fl)

    SEED = CFG["seed"]
    DEVICE = "cpu"

    # load model class dynamically
    BASE_MODEL_PATH = "model_classes"
    MODEL_CLASS_NAME = CFG["model"]["class"]
    spec = importlib.util.spec_from_file_location(MODEL_CLASS_NAME + '.py', os.path.join(BASE_MODEL_PATH, MODEL_CLASS_NAME + '.py'))
    ModelModule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ModelModule)
    ModelClass = ModelModule.DQN

    # seed
    random.seed(SEED)
    torch.manual_seed(SEED+1)

    # init env
    env = SushiGo(SEED+2, DEVICE)

    # init model
    model = ModelClass(
        len(Action), 
        len(Card),
        CARD_NUM,
        max(POSSIBLE_PLAYER_COUNTS),
        None,
        device=DEVICE)

    # load model
    state_dict = torch.load(SAVE_PATH)
    model.load_state_dict(state_dict)

    # build npcs
    l_all_npcs = list()
    for npc in CFG["npcs"]:
        new_npc = get_npc(npc["name"], **npc["kwargs"])
        l_all_npcs.append(new_npc)
