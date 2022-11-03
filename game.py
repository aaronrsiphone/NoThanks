import random
from itertools import groupby
from operator import itemgetter

class Player:
    """
    Base class for players in the game No Thanks.

    This particular player makes decisions randomly, 
    but new player strategies can be implementing the decide() function.
    """

    def __init__(self, player_number):
        # Set up empty hand
        self.hand = []
        # Store player number
        self.player_number = player_number

    def __str__(self):
        s = "Player: {}".format(self.player_number)
        s += "  Tokens: {}".format(self.tokens)
        s += "  Score: {}".format(self.score)
        s += " Hand: "
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

    @property 
    def state(self):
        d = {
            "hand": self.hand.copy(),
            "tokens" : self.tokens,
            "score" : self.score,
        }
        return d

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

class Human(Player):
    """
    This Player allows a human to play against the computers
    """
    def decide(self, game_state):

        print(game_state)
        choice = input("Do you want to take the card?")
        if choice.lower() in ["yes", "y", "true", "1"]:
            return True
        elif choice.lower() in ["no", "n", "false", "0"]:
            return False

class Denier(Player):
    """
    This player will always say "No Thanks" if able
    """
    def decide(self, _):
        if self.tokens < 0:
            return True

        return False

class Basic_math(Player):
    """
    This player inputs a threshold that is used to make a basic calculation.

    If the value of the card minus the tokens on the card is less than a value
    then the card will be taken
    """

    def __init__(self, player_number, threshold):
        super().__init__(player_number)
        self.threshold = threshold

    def decide(self, state):
        if self.tokens < 0:
            return True
        if state['flipped_card'] - state['tokens_on_card'] < self.threshold:
            return True

        return False
    
class Deck:
    """
    Class for the deck of cards in the game No Thanks
    """
    def __init__(self):
        # Create a deck with cards 3 through 35
        deck = list(range(3,36))
        # Shuffle the Deck
        random.shuffle(deck)
        # Discard the last nine cards
        self.deck = deck[:-9]
        # Start out the card with 0 tokens
        self.tokens = 0
        # Make an array for cards that were all ready taken
        self.taken = []

    @property
    def has_cards(self):
        return len(self.deck) > 0

    @property
    def flipped_card(self):
        if self.has_cards:
            return self.deck[-1]
        else:
            return 0

    def take_card(self):
        tokens = self.tokens
        card = self.deck.pop()
        self.tokens = 0
        self.taken.append(card)
        return (card, tokens)
    
class Game:
    """
    Class for the Game of No Thanks, contains all of the rules and actions
    """

    def __init__(self, players=[]):
        # Make a list of all the players
        self.players = players
        # Count the number of players
        self.n_players = len(self.players)
        # Deck setup
        self.deck = Deck()
        # Set up Turn Counter
        self.turn_counter = 0

        # Deal tokens to all the players
        n_tokens = {3:11, 4:11, 5:11, 6:9, 7:7}
        for player in self.players:
            player.tokens = n_tokens[self.n_players]
                
    @property
    def turn(self):
        return self.turn_counter%self.n_players

    @property
    def state(self):
        d = {
            "flipped_card" : self.deck.flipped_card,
            "tokens_on_card": self.deck.tokens,
            "player_states" : [p.state for p in self.players],
            "player_turn_index" : self.turn
        }
        return d
    
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
        player = self.players[self.turn]

        # Ask the player what they want to do
        choice = player.decide(self.state)

        # The player decides to take the card,
        # or is forced to because they don't have any tokens
        if choice or (player.tokens == 0):
            card, tokens = self.deck.take_card()
            player.hand.append(card)
            player.tokens += tokens
        # The player says "No Thanks!"
        else:
            player.tokens -= 1
            self.deck.tokens += 1
            self.turn_counter += 1

    def play_game(self):
        while self.deck.has_cards:
            self.player_action()

if __name__ == '__main__':
    
    p1 = Denier(1)
    p2 = Denier(2)
    p3 = Basic_math(3, 5)
    p4 = Basic_math(4, 0)

    game = Game(players=[p1, p2, p3, p4])
    winner = game.play_game()
    for p in game.players:
        print(p)
    game_state = game.state
    
