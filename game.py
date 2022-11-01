import random
from itertools import groupby
from operator import itemgetter

class Player:
    """
    Base class for players in the game No Thanks.

    This particular player makes decisions randomly, 
    but new player strategies can be implementing the decide() function.
    """

    def __init__(self, n_tokens, player_number):
        self.hand = []
        self._tokens = n_tokens
        self.player_number = player_number
        self.last_score = -1* n_tokens

    def __str__(self):
        s = "Player: {}".format(self.player_number)
        s += "  Tokens: {}".format(self.tokens)
        s += "  Score: {}".format(self.score)
        for seq in self.get_sequences():
            s += " {}".format(seq)

        return s

    @property
    def score(self):
        sequences = self.get_sequences()
        score = 0 
        for seq in sequences:
            score += seq[0]

        return score - self.tokens

    @property 
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, new_tokens):
        if new_tokens >= 0:
            self._tokens = new_tokens
        else:
            raise ValueError("User cannot have have negative tokens")

    def decide(self, game_state):
        """
        Function to decide if the player should take the card

        Return True to take the card

        Return False to say "No Thanks!"
        """
        n = random.randint(-1, 1)
        if n < 0:
            return False
        return True

    def has_neighbor(c):
        """
        checks if c will make a sequence with an existing card in hand. 
        """
        return (c - 1) in self.hand or (c + 1) in self.hand

    def get_sequences(self,c=None):
        
        test_hand = [h for h in self.hand]
        if c:
            test_hand.append(c)
        test_hand.sort()
        sequences = []
        for k, g in groupby(enumerate(test_hand), lambda i_x: i_x[0] - i_x[1]):
            sequences.append(sorted(list(map(itemgetter(1), g))))

        return sequences

    def get_state(self):

        state = {
            "hand": self.hand.copy(),
            "tokens" : self.tokens,
            "score" : self.calc_score(),
            "score_delta": self.calc_score() - self.last_score
        }
        return state

class Human(Player):

    def decide(self, game_state):

        print(game_state)

        choice = input("Do you want to take the card?")

        if choice.lower() in ["yes", "y", "true", "1"]:
            return True

        elif choice.lower() in ["no", "n", "false", "0"]:
            return False
        
    
class Deck:
    """
    Class for the deck of cards in the game No Thanks
    """
    def __init__(self):
        deck = list(range(3,36))
        random.shuffle(deck)
        self.deck = deck[:-9]
        self.dropped = deck[-9:]
        self.taken = []

        self.tokens = 0
    
    def get_flipped(self):
        if self.has_cards():
            return self.deck[-1]
        else:
            return 0

    def take_card(self):
        tokens = self.tokens
        card = self.deck.pop()

        self.tokens = 0
        self.taken.append(card)
        return (card, tokens)

    def has_cards(self):
        return len(self.deck) > 0
    
    def no_thanks(self):
        self.tokens += 1
    


class Game:
    """
    Class for the Game of No Thanks, contains all of the rules and actions
    """

    def __init__(self, players=[]):

        self.players = players
        self.n_players = len(self.players)
        
        # deck setup
        self.deck = Deck()
        self.current_player_index = 0
        self.turn_counter = 0

        self.game_log = []

    def get_turn_index(self):
        return self.turn_counter%self.n_players

    def get_state(self):
        player_scores = [(p.player_number, p.calc_score()) for p in self.players]
        player_scores.sort(key=lambda x : x[1])

        state = {
            "flipped_card" : self.deck.get_flipped(),
            "tokens_on_card": self.deck.tokens,
            "player_states" : [p.get_state() for p in self.players],
            "player_positions" : [p[0] for p in player_scores],
            "player_turn_index" : self.get_turn_index()
        }
        return state
    
    def player_action(self):
        """
        Function to allow the player to take an action

        Parameters
        ----------
        choice: Boolean
            If True, the player takes the card.
            If False, the player says "No Thanks!"
        """

        # Get the player whose turn it is
        player = self.players[self.get_turn_index()]
        # Ask the player what they want to do
        choice = player.decide(self.get_state())

        # The player decides to take the card
        if choice:
            card, tokens = self.deck.take_card()
            player.hand.append(card)
            player.tokens += tokens
        # The player says "No Thanks!"
        else:
            player.tokens -= 1
            self.deck.no_thanks()
            self.turn_counter +=1

    def play_game(self):
        while self.deck.has_cards():
            self.player_action()

if __name__ == '__main__':
    input("Ready?")
    
    p1 = Player(11, 1)
    p2 = Player(11, 2)
    p3 = Player(11, 3)
    p4 = Human(11, 4)

    game = Game(players = [p1, p2, p3, p4])
    winner = game.play_game()
    for p in game.players:
        print(p)
    game_state = game.get_state()
    
