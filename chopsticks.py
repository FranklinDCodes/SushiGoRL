
from agent import RLAgent
from game import SushiGo
from global_constants import *
import numpy as np

# plays chopsticks and updates the env, buffer, and agent
def use_chopsticks_agent(agent_game_state: np.array, agent: RLAgent, replay_buffer: MemoryBuffer, env: SushiGo):

    # get starting score
    last_agent_score = env.get_scores()[AGENT_TABLE_POS]

    # make copy of the state that is before the first chopstick choice
    state_pre_first_cs_choice = np.copy(agent_game_state)
    state_pre_first_cs_choice.table[Card.Chopsticks.value] -= 1

    # remove chopsticks from possible actions
    state_pre_first_cs_choice.possible_actions = [i for i in state_pre_first_cs_choice.possible_actions if i != Action.PlayChopsticks.value]

    # play chopsticks
    replay_buffer.push(Timestep(agent_game_state, action, state_pre_first_cs_choice, 0.0))
    
    # get first chopstick choice
    first_cs_choice = agent.select_action(state_pre_first_cs_choice)

    # check score before the move
    agent_last_score = env.get_scores()[AGENT_TABLE_POS]

    # play first choice
    env.play_chopsticks(AGENT_TABLE_POS, first_cs_choice)

    # get new state and remove chopsticks from possible actions
    state_pre_second_cs_choice = env.get_states(AGENT_TABLE_POS)
    state_pre_second_cs_choice.possible_actions = [i for i in state_pre_second_cs_choice.possible_actions if i != Action.PlayChopsticks.value]

    # add first chopstick pick to history
    first_choice_reward = env.get_scores()[AGENT_TABLE_POS] - agent_last_score
    replay_buffer.push(Timestep(state_pre_first_cs_choice, first_cs_choice, state_pre_second_cs_choice, first_choice_reward))

    # reset last score
    agent_last_score = env.get_scores()[AGENT_TABLE_POS]

    # get 2nd chopstick choice
    action = agent.select_action(state_pre_second_cs_choice)

    return action, state_pre_first_cs_choice, agent_last_score

# plays chopsticks and updates the env, and buffer for all npcs
def use_chopsticks_npc(npc_seat_numbers: np.array, l_played_chopsticks: list, tup_game_states: tuple, l_npcs: list, env: SushiGo):

    # get npc actions
    l_npc_actions = []
    for idx, npc_num in enumerate(npc_seat_numbers):

        # if npc played chopsticks
        if idx in l_played_chopsticks:

            # remove chopsticks from action choices
            tup_game_states[idx].possible_actions = [i for i in tup_game_states[idx].possible_actions if i != Action.Chopsticks.value]

            # get first chopstick choice
            first_choice = l_npcs[npc_num].select_action(tup_game_states[idx])
            env.play_chopsticks(idx, first_choice)

            # setup second choice options
            second_choice_state = env.get_states()[idx]
            second_choice_state.possible_actions = [i for i in second_choice_state.possible_actions if i != Action.Chopsticks.value]
            
            # get second choice
            second_choice = l_npcs[npc_num].select_action(second_choice_state)
            l_npc_actions[idx] = second_choice
    
    return np.array(l_npc_actions)

