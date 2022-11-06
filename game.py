import random

    
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
        for i in range(len(players)):
            self.players[i].assign_position(i)

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
        
        player_scores = [(p.player_number, p.score) for p in self.players]
        player_scores.sort(key=lambda x : x[1])
        
        return self.players[player_scores[0][0]]
