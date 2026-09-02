
# libraries
import random
import json
import os
import datetime
import importlib
import sys

# source code
from src.global_constants import *
from src.game import *
from src.epsilon import *
from src.agent import *
from src.npc import *
from src.chopsticks import *
from src.optimization import *
from src.metrics import *
from src.factories import *


# much of the code in this repo is modeled after the following tutorial
# some snippets are copied directly and edited, such as the MemoryBuffer class and Timestep object
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html



# CONFIG
config_name = sys.argv[1]
config_path = f"configs/{config_name}.json"

with open(config_path, 'r') as fl:
    CFG = json.load(fl)


# LOGGING AND SAVING
LOG_PATH = f"logs/{config_name}_{datetime.datetime.now().strftime('%m_%d_%Y_%H-%M-%S')}.log"
def log(s: any) -> None:

    print(s)
    with open(LOG_PATH, 'a') as fl:
        fl.write(str(s))



# HYPERPARAMETERS


ROUND_COUNT = CFG["episode_count"]
MAX_MEMORY = CFG["max_memory"]
BATCH_SIZE = CFG["batch_size"]
TAU = CFG["TAU"]
GAMMA = CFG["GAMMA"]
epsilon_func = get_epsilon_function(
    CFG["EPS_func"]["name"],
    **CFG["EPS_func"]["kwargs"]
)
SEED = CFG["seed"]



# CONSTANTS



def train_model():

    # seed
    random.seed(SEED)
    torch.manual_seed(SEED+1)

    # load model class dynamically
    BASE_MODEL_PATH = "src/model_classes"
    MODEL_CLASS_NAME = CFG["model"]["class"]
    spec = importlib.util.spec_from_file_location(MODEL_CLASS_NAME + '.py', os.path.join(BASE_MODEL_PATH, MODEL_CLASS_NAME + '.py'))
    ModelModule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ModelModule)
    ModelClass = ModelModule.DQN

    # device
    GAME_DEVICE = CFG["game"]["device"]
    MODEL_DEVICE = CFG["model"]["device"]

    RESULTS_DIR = CFG.get("output_dir", DEFAULT_RESULTS_DIR)

    # init env
    env = SushiGo(SEED+2, GAME_DEVICE)

    # init model
    lr_scheduler = get_scheduler(CFG.get("lr"))
    optimizer = get_optimizer_class(CFG.get("optimizer"))
    loss = get_loss(CFG.get("loss"))
    model = ModelClass(
        len(Action), 
        len(Card),
        CARD_NUM,
        max(POSSIBLE_PLAYER_COUNTS),
        lr_scheduler,
        optimizer,
        loss,
        device=MODEL_DEVICE)

    # build agent
    replay_buffer = MemoryBuffer(CFG["max_memory"])
    agent = RLAgent(model, epsilon_func, SEED+3)

    # build npcs
    l_all_npcs = list()
    for npc in CFG["npcs"]:
        new_npc = get_npc(npc["name"], **npc["kwargs"])
        l_all_npcs.append(new_npc)

    start = datetime.datetime.now()

    # make directory for output
    dir = f"{RESULTS_DIR}/{config_name}_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    model_save_dir = f"{dir}/model_saves/"
    os.makedirs(model_save_dir)

    # metrics tracker
    metric_tracker = Metrics()

    for ep in range(ROUND_COUNT):

        # get player count
        player_count = random.choice(POSSIBLE_PLAYER_COUNTS)

        # random select npcs and order
        random.shuffle(l_all_npcs)
        l_game_npcs = l_all_npcs[:player_count - 1]

        # setup game
        # kind of has some unnecessary computation in this function for some of the rounds
        env.setup_new_game(player_count)

        # get starting env information
        tup_game_states = env.get_states()
        agent_game_state = tup_game_states[AGENT_TABLE_POS]
        npc_game_states = [i for idx, i in enumerate(tup_game_states) if idx != AGENT_TABLE_POS]

        # get starting scores
        last_agent_score = 0

        # take steps in episode
        episode_over = False
        while not episode_over:

            # get agent action
            action = agent.select_action(agent_game_state)

            # get npc actions
            l_npc_actions = []
            for i in range(player_count - 1):
                npc_state = npc_game_states[i]
                l_npc_actions.append(l_game_npcs[i].select_action(npc_state))
            t_npc_actions = torch.tensor(l_npc_actions, device=GAME_DEVICE)

            # check if agent used chopsticks
            chopsticks_played = action == Action.PlayChopsticks.value
            if chopsticks_played:
                
                # call chopsticks function
                action, agent_game_state, last_agent_score = use_chopsticks_agent(
                    agent_game_state,
                    agent,
                    replay_buffer,
                    env
                )

            # check for npc chopsticks
            t_npc_used_chopsticks = t_npc_actions == Action.PlayChopsticks.value
            l_played_chopsticks = torch.arange(player_count - 1)[t_npc_used_chopsticks.cpu()].tolist()
            if torch.any(t_npc_used_chopsticks):

                # call function to manage npc chopsticks use
                t_npc_actions = use_chopsticks_npc(
                    t_npc_actions,
                    l_played_chopsticks,
                    npc_game_states,
                    l_game_npcs,
                    env
                )

            # take actions
            t_all_player_cards = torch.tensor([action, *t_npc_actions], device=GAME_DEVICE)
            env.play_cards(t_all_player_cards)

            # pass hands
            env.pass_hands()

            # check if round is over to determine if there is a real resulting state (for the memory)
            if env.round_is_over():

                new_agent_game_state = None

            else:

                # get new resulting states
                tup_new_game_states = env.get_states() 
                new_agent_game_state = tup_new_game_states[0]

            # get rewards
            agent_score = env.get_agent_round_score()
            reward = agent_score - last_agent_score

            # save history
            replay_buffer.push(agent_game_state, action, new_agent_game_state, reward)

            # setup new round
            if env.round_is_over() and env.round_num != 3:
                
                env.setup_new_round(player_count)

                # get game states
                tup_new_game_states = env.get_states()
                new_agent_game_state = tup_new_game_states[0]
                agent_score = 0

            # end game
            elif env.round_is_over() and env.round_num == 3:
                episode_over = True

            # set old states
            tup_game_states = tup_new_game_states
            agent_game_state = new_agent_game_state
            last_agent_score = agent_score
            npc_game_states = [i for idx, i in enumerate(tup_game_states) if idx != AGENT_TABLE_POS]

            # optimize
            loss = optimize(BATCH_SIZE, replay_buffer, agent.policy_net, agent.target_net, GAMMA, ep, GAME_DEVICE)
            metric_tracker.aggregate('losses', loss)

            # update target net
            agent.soft_update_target_net(TAU)

        # update metrics
        env.end_game()
        metric_tracker.aggregate('npc_score', env.get_game_scores())
        metric_tracker.aggregate('scores', env.get_game_scores()[AGENT_TABLE_POS].cpu().item())

        if (ep+1) % 1000 == 0:
            print(f"Time to {ep+1}: {datetime.datetime.now() - start}")
            print(f"{agent.update_count} updates completed")
            start = datetime.datetime.now()
            metric_tracker.commit_current_to_history()
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_save_{ep+1}.pkl"))

    metric_tracker.save_to_txt(os.path.join(dir, "stats.txt"))


if __name__ == "__main__":
    train_model()

