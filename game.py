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
        # set up empty hand
        self.hand = []
        # store player number
        self.player_number = player_number

    def __str__(self):
        s = "player: {}".format(self.player_number)
        s += "  tokens: {}".format(self.tokens)
        s += "  score: {}".format(self.score)
        s += " hand: "
        for seq in self.sequences:
            s += " {}".format(seq)

        return s

    @property
    def score(self):
        score = 0 
        for seq in self.sequences:
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
            raise valueerror("user cannot have have negative tokens")
    
    @property
    def sequences(self):
        """
        Returns the hand sorted by sequences
        """
        test_hand = self.hand.copy()
        test_hand.sort()
        sequences = []
        for k, g in groupby(enumerate(test_hand), lambda i_x: i_x[0] - i_x[1]):
            sequences.append(sorted(list(map(itemgetter(1), g))))

        return sequences

    @property 
    def state(self):
        d = {
            "index" : self.player_number,
            "hand": self.hand.copy(),
            "tokens" : self.tokens,
            "score" : self.score,
        }
        return d

    def decide(self, game_state):
        """
        Function to decide if the player should take the card

        Return true to take the card

        Return false to say "no thanks!"
        """
        n = random.randint(-1, 1)
        if n < 0:
            return false
        return true

    def test_card(self, card):
        """
        Function to test a card and decide what the impact on 
        the score would be
        """
        # Save the state of the hand
        old_hand = self.hand.copy()
        # Save the current score
        old_score = self.score
        # Temporarily add the card to the hand
        self.hand.append(card)
        # Calculate a new score
        new_score = self.score
        # Return the hand to the state before the test card was added
        self.hand = old_hand

        # Return the difference in score
        return new_score - old_score

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

class Net_score(Basic_math):
    """
    This player calculates what the net score impact would be of taking the card.

    If the net score is less than the set threshold, then the card will be taken
    """

    def decide(self, state):
        if self.tokens < 0:
            return True
        
        card = state['flipped_card']
        tokens = state['tokens_on_card']
        score_delta = self.test_card(card) - tokens

        if score_delta < self.threshold:
            return True

        return False

class AaronsRules(Player):

    def __init__(self, 
                player_number, 
                auto_take_threshold=5,
                general_threshold=5):

        super().__init__(player_number)
        self.auto_take_threshold = auto_take_threshold
        self.general_threshold = general_threshold
        

    def will_make_sequence(self, c, hand=None):

        if not hand:
            hand = self.hand
        return (c - 1) in hand or (c + 1) in hand

    def decide(self, state):
         # Returning -1 is Nothanks
        # Returning 1 is to take.
        
        # if you dont have any tokes, take it
        if self.tokens == 0:
            return 1

        card = state['flipped_card']
        if card < self.auto_take_threshold:
            return 1
        
        valuable = self.will_make_sequence(card)
        valuable_to_others = False
        
        for p in state['player_states']:
            if p['index'] != self.player_number:
                if self.will_make_sequence(card, p['hand']):
                    valuable_to_others = True

        number_of_players = len(state["player_states"])
        let_it_ride_token_max = 2*number_of_players

        if valuable:
            # if valuable to you and others, take it first. 
            if valuable_to_others:
                return 1
            # if there is a risk others will take it on the next go around, take it early
            if card - state['tokens_on_card'] - number_of_players < self.general_threshold:
                return 1
            else:
                # if valuable only to you, let it ride twice. 
                if state['tokens_on_card'] >= let_it_ride_token_max:
                    return 1
        # If value of card minus n tokens on card is less than a threshold, take it.
        elif card - state['tokens_on_card'] < self.general_threshold:
            return 1

        return -1

    
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
            for p in self.players:
                print(p)
            self.player_action()
            print(self.deck.deck)
        
        player_scores = [(p.player_number, p.score) for p in game.players]
        player_scores.sort(key=lambda x : x[1])
        
        return player_scores[0][0]

if __name__ == '__main__':
    
    p1 = Denier(0)
    p2 = Basic_math(1, 10)
    p3 = Net_score(2, 3)
    p4 = AaronsRules(3, 5,5)

    winners = [0,0,0,0]
    for _ in range(1):
        game = Game(players=[p1, p2, p3,p4])
        winner = game.play_game()
        winners[winner] += 1
    
    print(winners)
    
