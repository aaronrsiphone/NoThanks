from game import Game
from players import *
import random

if __name__ == "__main__":

    players = [
        Human(),
        LetItRider(21,13),
        NetScore(12),
        BasicMath(12)]

    random.shuffle(players)

    game = Game(players)

    winner = game.play_game()
    for player in game.players:
        print(player)
    