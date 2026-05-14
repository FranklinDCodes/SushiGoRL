import random
from game import SushiGo, Card
import numpy as np


game = SushiGo(4, 87)
print(game.hands.cards_left)



# EMPTY GAME
print(game)


# TURN 1
game.play_cards(np.array([
    Card.Tempura.value, 
    Card.Dumpling.value, 
    Card.Maki_2.value, 
    Card.Egg_Nigiri.value
]))
game.pass_hands()
print(game)


# TURN 2
game.play_cards(np.array([
    Card.Sashimi.value, 
    Card.Wasabi.value, 
    Card.Dumpling.value, 
    Card.Chopsticks.value
]))
game.pass_hands()
print(game)


# TURN 3 : USING CHOPSTICKS
game.play_cards(np.array([
    Card.Tempura.value, 
    Card.Dumpling.value, 
    Card.Dumpling.value, 
    Card.Maki_1.value
]))

# PRE CHOPSTICKS
print(game)

# POST CHOPSTICKS
game.play_chopsticks(3, Card.Maki_1.value)
print(game)

# PASS
game.pass_hands()
print(game)


# TURN 4 : USING WASABI
game.play_cards(np.array([
    Card.Chopsticks.value, 
    Card.Egg_Nigiri.value, 
    Card.Pudding.value, 
    Card.Maki_3.value
]))
game.pass_hands()
print(game)


# TURN 5
game.play_cards(np.array([
    Card.Pudding.value, 
    Card.Salmon_Nigiri.value, 
    Card.Maki_3.value, 
    Card.Tempura.value
]))
game.pass_hands()
print(game)


# TURN 6
game.play_cards(np.array([
    Card.Wasabi.value, 
    Card.Tempura.value, 
    Card.Sashimi.value, 
    Card.Pudding.value
]))
game.pass_hands()
print(game)


# TURN 7 : CHOPSTICKS AND WASABI
game.play_cards(np.array([
    Card.Tempura.value, 
    Card.Tempura.value, 
    Card.Sashimi.value, 
    Card.Wasabi.value
]))

# PRE CHOPSTICKS
print(game)

# POST CHOPSTICKS
game.play_chopsticks(0, Card.Salmon_Nigiri.value)
print(game)

# PASS
game.pass_hands()
print(game)


# TURN 8 : END
game.play_cards(np.array([
    Card.Sashimi.value, 
    Card.Sashimi.value, 
    Card.Chopsticks.value, 
    Card.Chopsticks.value
]))
game.pass_hands()
print(game)
print(game.get_scoreboard())
game.setup_new_round()



# round 2

for i in range(8):

    states = game.get_states()

    game.play_cards(np.array([
        int(random.choice(np.arange(states[0].hand.shape[0])[states[0].hand != 0])),
        int(random.choice(np.arange(states[0].hand.shape[0])[states[1].hand != 0])),
        int(random.choice(np.arange(states[0].hand.shape[0])[states[2].hand != 0])),
        int(random.choice(np.arange(states[0].hand.shape[0])[states[3].hand != 0]))
    ]))
    game.pass_hands()

print(game)
print(game.get_scoreboard())
game.setup_new_round()



# round 3

for i in range(8):

    states = game.get_states()

    game.play_cards(np.array([
        int(random.choice(np.arange(states[0].hand.shape[0])[states[0].hand != 0])),
        int(random.choice(np.arange(states[0].hand.shape[0])[states[1].hand != 0])),
        int(random.choice(np.arange(states[0].hand.shape[0])[states[2].hand != 0])),
        int(random.choice(np.arange(states[0].hand.shape[0])[states[3].hand != 0]))
    ]))
    game.pass_hands()

print(game)
print(game.get_scoreboard())



