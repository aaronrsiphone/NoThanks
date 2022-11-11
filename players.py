from itertools import groupby
from operator import itemgetter
import random

class Player:
    """
    Base class for players in the game No Thanks.

    This particular player makes decisions randomly, 
    but new player strategies can be implementing the decide() function.
    """

    def __init__(self):
        # set up empty hand
        self.hand = []
        # store player number
        
    
    def assign_position(self, player_number):
        self.player_number = player_number

    def __str__(self):
        s = "player: {}".format(self.player_number)
        s += "  type: {}".format(type(self))
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
            raise ValueError("user cannot have have negative tokens")
    
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
            "hand": sorted(self.hand.copy()),
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
            return False
        return True

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

        print(f"Its Your turn Player {self.player_number}")
        print(f"You have {self.tokens}")
        
        for ps in game_state['player_states'][self.player_number+1:] + game_state['player_states'][:self.player_number]:
            print(f"{ps['index']} hand is {sorted(ps['hand'])}")

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

class BasicMath(Player):
    """
    This player inputs a threshold that is used to make a basic calculation.

    If the value of the card minus the tokens on the card is less than a value
    then the card will be taken
    """

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def decide(self, state):
        if self.tokens < 0:
            return True
        if state['flipped_card'] - state['tokens_on_card'] < self.threshold:
            return True

        return False

class NetScore(BasicMath):
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
    """
    This player denies cards until it is forced to take one. 
    This only accepts cards that complement its current hand taking into account the number of tokens on the card
    and if the card is valuable to other players.
    """
    def __init__(self, 
                auto_take_threshold=5,
                general_threshold=5):

        super().__init__()
        self.auto_take_threshold = auto_take_threshold
        self.general_threshold = general_threshold
        

    def will_make_sequence(self, c, hand=None):

        if not hand:
            hand = self.hand
        one = (c - 1) in hand
        two = (c + 1) in hand
        return  one or two

    def decide(self, state):
         # Returning -1 is Nothanks
        # Returning 1 is to take.
        
        # if you dont have any tokes, take it
        if self.tokens == 0:
            return True

        card = state['flipped_card']
        
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
                return True
            # if there is a risk others will take it on the next go around, take it early
            if card - state['tokens_on_card'] + number_of_players < self.general_threshold:
                return True
            else:
                # if valuable only to you, let it ride twice. 
                if card-state['tokens_on_card'] >= let_it_ride_token_max:
                    return True
        else:
            if card - state['tokens_on_card'] < self.auto_take_threshold:
                return True

        return False


class LetItRider(Player):
    """ 
    This player takes a high card with many tokens as its first card and
    only takes cards that complement its hand sequentially once the number
    of tokens on said card meets a token threshold
    """

    def __init__(self, card_threshold=25, tokens_threshold=8):
        super().__init__()
        self.card_threshold = card_threshold
        self.tokens_threshold = tokens_threshold

    def will_make_sequence(self, c, hand=None):

        if not hand:
            hand = self.hand
        one = (c - 1) in hand
        two = (c + 1) in hand
        return  one or two

    def decide(self, state):
        # Returning True is take
        # Returning False is pass
        
        card = state['flipped_card']
        tokens_on_card = state['tokens_on_card']

        if len(self.hand) == 0:
            if card >= self.card_threshold and tokens_on_card > self.tokens_threshold:
                return True
        else:
            valuable = self.will_make_sequence(card)
            if valuable and tokens_on_card > self.tokens_threshold:
                return True
        
        return False

class SmartLetItRider():

    """ 
    This player takes a high card with many tokens as its first card and
    only takes cards that complement its hand sequentially once the number
    of tokens on said card meets a token threshold
    """

    def __init__(self, card_threshold=20, card_tokens_threshold=13, valuable_token_threshold=13, lt_card_threshold=20, lt_card_tokens_threshold=13, lt_tokens_threshold=0):
        super().__init__()
        self.card_threshold = card_threshold
        self.card_tokens_threshold = card_tokens_threshold
        self.valuable_token_threshold = valuable_token_threshold
        self.lt_card_threshold = lt_card_threshold
        self.lt_card_tokens_threshold = lt_card_tokens_threshold
        self.lt_tokens_threshold = lt_tokens_threshold

    def will_make_sequence(self, c, hand=None):

        if not hand:
            hand = self.hand
        one = (c - 1) in hand
        two = (c + 1) in hand
        return  one or two

    def decide(self, state):
        # Returning True is take
        # Returning False is pass
        
        card = state['flipped_card']
        tokens_on_card = state['tokens_on_card']

        if len(self.hand) == 0:
            if card >= self.card_threshold and tokens_on_card > self.card_tokens_threshold:
                return True
        else:
            valuable = self.will_make_sequence(card)
            if valuable and tokens_on_card > self.valuable_token_threshold:
                return True
        
        # Low Tokens
        if self.tokens < self.lt_token_threshold:
            if card < self.lt_card_threshold and tokens_on_card > self.lt_card_tokens_threshold:
                return True

        return False

if __name__ == "__main__":
    pass