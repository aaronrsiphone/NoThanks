from game import Game
from players import *

import random

mine = "LetItRider"

win_count_by_player_class = {}
winners = [0,0,0,0,0]
class_scores = {}
n_runs = 1000

for index in range(n_runs):

    # if (index+1)%100 == 0:
        # print(index)
        # print(win_count_by_player_class)
        
    # Players were optimized by parameter sweep
    players = [
        AaronsRules(12,16), # Optimized to be 12, 16
        NetScore(12), # Optimized to be 12 
        BasicMath(12), # Optimized to be 12
        LetItRider(24,14), # Optimized to be 24, 14
        Denier()
    ]

    random.shuffle(players)
    game = Game(players=players)
    winner = game.play_game()
    
    winning_class = winner.__class__.__name__
    winners[winner.player_number] +=1
    if winning_class in win_count_by_player_class:
        win_count_by_player_class[winning_class] += 1
    else:
        win_count_by_player_class[winning_class] = 1

    for player in players:
        if player.__class__.__name__ not in class_scores:
            class_scores[player.__class__.__name__] = 0
        class_scores[player.__class__.__name__] += player.score

for k, v in class_scores.items():
    mean = v/n_runs
    print(k, mean, win_count_by_player_class[k])
        # for k, v in win_count_by_player_class.items():
        #     print(f"{k} won {v} games")
        # print(winners)
            
# BasicMath(12) position Bias
# [25967, 24700, 24067, 25266]
# [25980, 24938, 23969, 25113]

# NetScore(12) Position Bias
# [26792, 25681, 24453, 23074]
# [26673, 25897, 24391, 23039]

# AaronsRules(12,16) Position Bias
# [27208, 25575, 24104, 23113]
# [26889, 25514, 24531, 23066]
# [26906, 25639, 24296, 23159]

# Denier() PositionBias
# [99996, 3, 1, 0]
# [99999, 1, 0, 0]