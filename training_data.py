import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

from game import *
import csv
import sys


FILE_NUMBER = sys.argv[1]

# Globals
NEG_ENCODE = -1
POS_ENCODE = 1

def encode_cards(cards):
    deck_encoding = [NEG_ENCODE]*33
    for c in cards:
        if c != 0:
            deck_encoding[c-3] = POS_ENCODE
    return deck_encoding


def encode_cards(cards):
    deck_encoding = [NEG_ENCODE]*33
    for c in cards:
        if c != 0:
            deck_encoding[c-3] = POS_ENCODE
    return deck_encoding

def token_encoding(tokens):
    encoding = [NEG_ENCODE]*44
    if tokens > 0:
        encoding[tokens] = POS_ENCODE
    return encoding

def encode_position(p_index, positions):
    encodeing = [NEG_ENCODE]*N_PLAYERS
    encodeing[positions.index(p_index)] = POS_ENCODE
    return encodeing

def encode_state(state, player_state=True, all_player_states=False, player_rank= False):

    flipped_card_encoding = encode_cards([state['flipped_card']])
    tokens_on_card_encoding = token_encoding(state['tokens_on_card'])
 
    encoded_state = flipped_card_encoding + tokens_on_card_encoding

    if player_state or all_player_states:

        player_card_encoding = []
        player_token_encoding = []
        players = reorder(state['player_states'], state["player_turn_index"])
        for p in players:
            player_card_encoding.extend(encode_cards(p['hand']))
            player_token_encoding.extend(token_encoding(p['tokens']))
            if not all_player_states:
                break
        
        encoded_state.extend(player_card_encoding + player_token_encoding)
    
    if player_rank:
        player_rank_encoding = encode_position(state["player_turn_index"], state['player_positions'])
        encoded_state.extend(player_rank_encoding)

    return encoded_state
    
def reorder(the_list, first):
    return the_list[first:] + the_list[:first]

def play_random_game(game):
    game_log = []
    action_log = []
    score_changes = []
    while game.deck.has_cards():
        game_log.append(game.get_state())
        action = random.choice([-1,1])
        action_log.append(action)
        delta_score = game.player_action(action)
        score_changes.append(delta_score)


    # who wins
    player_scores = [(p.player_number, p.calc_score()) for p in game.players]
    player_scores.sort(key=lambda x : x[1])

    return player_scores, game_log, action_log, score_changes


def play_one_random_game():
    game = Game(N_PLAYERS)
    players_and_scores, game_states, actions, score_changes = play_random_game(game)
    encoded_game_data = []
    player_odering = [p[0] for p in players_and_scores] # contains list of players in order of score [0,2,1,3], player 1 is in third place. 
    for row, action, score_change in zip(game_states, actions, score_changes):
        encoded_row = encode_state(row, player_state=True, all_player_states=True, player_rank= True)
        encoded_row.append(action)
        index = player_odering.index(row["player_turn_index"])
        encoded_row.append(players_and_scores[index][1])
        encoded_row.append(score_change)

        encoded_game_data.append(encoded_row)

    return encoded_game_data


def play_one_random_game2():
    game = Game(N_PLAYERS)
    players_and_scores, game_states, actions, score_changes = play_random_game(game)
    encoded_game_data = []
    
    for row, action, score_change in zip(game_states, actions, score_changes):
        encoded_row = encode_state(row, player_state=True, all_player_states=False, player_rank=False)
        encoded_row.append(action)
        encoded_row.append(score_change)

        encoded_game_data.append(encoded_row)

    return encoded_game_data


def squared_error_masked(y_true, y_pred):
    err = K.sum(y_true * y_pred)
    return err

# Need to rewrite this whole thing to ask the player what their next move is. 
# It makes it easier to override the players behavior model
def play_game_with_model(games,model, player_index = 0):
    game_log = []
    action_log = []
    score_changes = []
    game_encodings = []
    prediction_game_indexes = []

    # for i, game in enumerate(games):
    #     state = game.get_state()
    #     if state["player_turn_index"] == player_index:
    #         model_input = encode_state(game.get_state(), player_state=True, all_player_states=False, player_rank=False)
    #         game_encodings.append(model_input)
    
    # predictions = model.predict(np.array(game_encodings))
    active_games = len(games)

    while active_games > 0:
        games_needing_predictions =  []
        prediction_inputs = []
        for game in games:
            if game.deck.has_cards():
                state = game.get_state()
                state['player_states'] = reorder(state['player_states'],state["player_turn_index"])
                game_log.append(state)
                action = 0
                if state["player_turn_index"] == player_index:
                    model_input = encode_state(state, player_state=True, all_player_states=False, player_rank=False)
                    games_needing_predictions.append(game)
                    prediction_inputs.append(model_input)

                    # model_choice = model.predict(np.array([model_input,]))
                    # action = model_choice.argmax()*2 -1
                    # action_log.append(action)
                    # delta_score = game.player_action(action)
                    # score_changes.append(delta_score)
                else:
                    action = random.choice([-1,1])
                    game.player_action(action)

                if not game.deck.has_cards():
                    active_games -= 1

        if len(prediction_inputs) > 0:
            predictions = model.predict(np.array(prediction_inputs))
            for pred, game in zip(predictions, games_needing_predictions):

                game_log.append(game.get_state())

                player = game.players[game.get_turn_index()]
                print(player.get_state())
                if player.tokens > 0:
                    action = pred.argmax()*2 -1
                    action_log.append(action)
                else:
                    action = -1
                    action_log.append(0)
                delta_score = game.player_action(action)
                score_changes.append(delta_score)

                if not game.deck.has_cards():
                    active_games -= 1
            print("Active Games: ", active_games)


    # who wins
    player_scores = [(p.player_number, p.calc_score()) for p in game.players]
    player_scores.sort(key=lambda x : x[1])

    return player_scores, game_log, action_log, score_changes


def play_one_game_with_model(model_path):
    games = [Game(N_PLAYERS) for _ in range(1)]


    model = keras.models.load_model(model_path,custom_objects={"squared_error_masked": squared_error_masked})

    players_and_scores, game_states, actions, score_changes = play_game_with_model(games, model)
    encoded_game_data = []
    
    for row, action, score_change in zip(game_states, actions, score_changes):
        encoded_row = encode_state(row, player_state=True, all_player_states=False, player_rank=False)
        encoded_row.append(action)
        encoded_row.append(score_change)

        encoded_game_data.append(encoded_row)

    return encoded_game_data


def dump_data(data):
    with open(f'random_game_play_{N_GAMES}_games_{FILE_NUMBER}.csv', 'a') as file:
        for game in data:
            for row in game:
                file.write(','.join([str(v) for v in row]))
                file.write('\n')



if __name__ == '__main__':
    N_GAMES = 1

    data = []
    counter = 0
    for i in range(N_GAMES):
        
        if (i+1)%1000 == 0:
            print("Run: ", i+1, "of ", N_GAMES)

        counter += 1
        # data.append(play_one_random_game2())
        data.append(play_one_game_with_model("C:\\Users\\renfroe\\Dev\\NoThanks\\models\\model_0"))

        if counter == 1000:
            print("Saving")
            dump_data(data)
            data = []
            counter = 0

    if len(data)> 0:
        dump_data(data)
    
    # for p in game.players:
    #     p.print_info()
    # game_state = game.get_state()
    
    # print(encode_state(game_state))
    