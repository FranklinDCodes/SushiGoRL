from npc import *
from game import *


game = SushiGo(4, seed=14)

agent_rand = RandomNPC()

for i in range(8):

    states = game.get_states()

    game.play_cards(np.array([
        agent_rand.select_action(states[0]),
        agent_rand.select_action(states[1]),
        agent_rand.select_action(states[2]),
        agent_rand.select_action(states[3])
    ]))
    game.pass_hands()

    print(game)




