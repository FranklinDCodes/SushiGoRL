
from agent import RLAgent
from game import SushiGo
from global_constants import *
import torch
from copy import deepcopy

# plays chopsticks and updates the env, buffer, and agent
def use_chopsticks_agent(agent_game_state: PlayerState, agent: RLAgent, replay_buffer: MemoryBuffer, env: SushiGo):

    # make copy of the state that is before the first chopstick choice
    new_state_table = torch.clone(agent_game_state.table)
    new_state_table[AGENT_TABLE_POS, Card.Chopsticks.value] -= 1
    new_state_possible_actions = torch.tensor([i for i in agent_game_state.possible_actions if i != Action.PlayChopsticks.value], device=env.device)
    state_pre_first_cs_choice = PlayerState(
        AGENT_TABLE_POS,
        agent_game_state.hand,
        new_state_table,
        new_state_possible_actions
    )

    # push to history if buffer
    if replay_buffer is not None:
        replay_buffer.push(agent_game_state, Action.PlayChopsticks.value, state_pre_first_cs_choice, 0.0)

    # get first chopstick choice
    first_cs_choice = agent.select_action(state_pre_first_cs_choice)

    # check score before the move
    agent_last_score = env.get_agent_round_score()

    # play first choice
    # puts chopstick back in hand and off of table
    env.play_chopsticks(AGENT_TABLE_POS, first_cs_choice)

    # get new state and remove chopsticks from possible actions
    new_agent_state = env.get_states(AGENT_TABLE_POS)[0]
    state_pre_second_cs_choice = PlayerState(
        AGENT_TABLE_POS,
        new_agent_state.hand,
        new_agent_state.table,
        torch.tensor([i for i in new_agent_state.possible_actions if i != Action.PlayChopsticks.value and i != Action.Chopsticks.value], device=env.device)
    )

    first_choice_reward = env.get_agent_round_score() - agent_last_score

    # push to history if buffer
    if replay_buffer is not None:
        replay_buffer.push(state_pre_first_cs_choice, first_cs_choice, state_pre_second_cs_choice, first_choice_reward)

    # reset last score
    agent_last_score = env.get_agent_round_score()

    # get 2nd chopstick choice
    action = agent.select_action(state_pre_second_cs_choice)

    return action, state_pre_first_cs_choice, agent_last_score

# plays chopsticks and updates the env, and buffer for all npcs
def use_chopsticks_npc(t_npc_actions_played: torch.Tensor, l_played_chopsticks: list, tup_game_states: tuple, l_npcs: list, env: SushiGo):

    # get npc actions
    l_npc_actions = []
    for idx, state in enumerate(tup_game_states):

        # if npc played chopsticks
        if idx in l_played_chopsticks:

            # remove chopsticks from action choices
            state_for_first_choice = PlayerState(
                idx,
                state.hand,
                state.table,
                torch.tensor([i for i in state.possible_actions if i != Action.PlayChopsticks.value], device=env.device)
            )

            if len(state_for_first_choice.possible_actions) == 0: # DEBUG
                raise Exception("No possible actions")

            # get first chopstick choice
            # player id is raw idx + 1 because the agent is idx 0
            first_choice = l_npcs[idx].select_action(state_for_first_choice)
            env.play_chopsticks(idx+1, first_choice)

            # setup second choice options
            new_state = env.get_states()[idx+1]
            state_for_second_choice = PlayerState(
                idx,
                new_state.hand,
                new_state.table,
                torch.tensor([i for i in new_state.possible_actions if i != Action.PlayChopsticks.value and i != Action.Chopsticks.value], device=env.device)
            )

            if len(state_for_second_choice.possible_actions) == 0: # DEBUG
                raise Exception("No possible actions")
            
            # get second choice
            second_choice = l_npcs[idx].select_action(state_for_second_choice)
            t_npc_actions_played[idx] = second_choice
    
    return t_npc_actions_played

