from game import Game
from players import *

import random
import itertools
from multiprocessing import Pool, freeze_support

def run(player_class, player_class_args, other_player_classes_and_params, n_runs, shuffle=True):
    
    win_count_by_player_class = {player_class.__name__ : 0}
    for p in other_player_classes_and_params:
        win_count_by_player_class[p[0].__name__] = 0

    for index in range(n_runs):
        players = [p[0](*p[1]) for p in other_player_classes_and_params]
        players.append(player_class(*player_class_args))
        if shuffle:
            random.shuffle(players)
        
        game = Game(players=players)
        winner = game.play_game()
        
        winning_class = winner.__class__.__name__

        win_count_by_player_class[winning_class] += 1
    
    for k, v in win_count_by_player_class.items():
        win_count_by_player_class[k] = v/n_runs
    
    return {"params" : player_class_args,
            "scores" : win_count_by_player_class}


def test_player(player_class, parameter_ranges, other_player_classes_and_params, n_runs, threads=16, shuffle=True):
    """
    Given a player class to test,
          a list of parameter ranges as tuples,
          a list of other player classes and instantiation parameters,
          and number of runs

    Returns an array of dictionaries with the probability of each player winning for each parameter set passed to the player class under test.

    Example: This will instantiate LetItRider with all combinations args. First arg being between 5 and 29 and second arg being between 10 and 15. 
            Running 5000 games with each parametric combination. 
    .. 
    >>> others = [
        (BasicMath,   (12,)),
        (NetScore,    (12,)),
        (AaronsRules, (12,16)),
       ]
    >>> results = scratch.test_player(LetItRider, [(5,29),(10,15)], others, 5000)
    >>> print(result[0])
    [{'params': (5, 10), 'scores': {'LetItRider': 0.76, 'NetScore': 0.12, 'AaronsRules': 0.12}},
    """
    test_class_name = player_class.__name__
    parameter_sets =  itertools.product(*[range(p[0], p[1]) for p in parameter_ranges])

    map_params = []
    for ps in parameter_sets:
        map_params.append([player_class, ps, other_player_classes_and_params,n_runs,shuffle])

    with Pool(threads) as p:
        results = p.starmap(run, map_params)
    
    return results


# for k, v in class_scores.items():
#     mean = v/n_runs
#     print(k, mean, win_count_by_player_class[k])

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