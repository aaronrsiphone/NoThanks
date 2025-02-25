import random
import time
import json
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any

class Deck:
    """
    Class for the deck of cards in the game No Thanks
    """
    def __init__(self):
        # Create a deck with cards 3 through 35
        deck = list(range(3, 36))
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
    
    def serialize(self) -> Dict:
        """Serialize the deck state to a dictionary"""
        return {
            "deck": self.deck.copy(),
            "tokens": self.tokens,
            "taken": self.taken.copy(),
            "flipped_card": self.flipped_card
        }
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'Deck':
        """Create a deck from serialized data"""
        deck = cls()
        deck.deck = data["deck"]
        deck.tokens = data["tokens"]
        deck.taken = data["taken"]
        return deck

class Game:
    """
    Class for the Game of No Thanks, contains all of the rules and actions.
    Enhanced for ML compatibility.
    """

    def __init__(self, players=[], verbose=False, seed=None):
        # Set random seed if provided for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Make a list of all the players
        self.players = players
        self.verbose = verbose
        for i in range(len(players)):
            self.players[i].assign_position(i)

        # Count the number of players
        self.n_players = len(self.players)
        # Deck setup
        self.deck = Deck()
        # Set up Turn Counter
        self.turn_counter = 0
        # Track game history for ML training
        self.history = []
        # Track rewards over time
        self.rewards_history = [0] * self.n_players
        # Initial scores
        self.initial_scores = [0] * self.n_players

        # Deal tokens to all the players
        n_tokens = {3: 11, 4: 11, 5: 11, 6: 9, 7: 7}
        for player in self.players:
            player.tokens = n_tokens[self.n_players]
            
        # Record initial state
        self._record_state()
                
    @property
    def turn(self):
        return self.turn_counter % self.n_players

    @property
    def state(self) -> Dict:
        """
        Return the full game state in a structured format
        """
        d = {
            "flipped_card": self.deck.flipped_card,
            "tokens_on_card": self.deck.tokens,
            "player_states": [p.state for p in self.players],
            "player_turn_index": self.turn,
            "cards_remaining": len(self.deck.deck),
            "cards_taken": self.deck.taken,
            "turn_counter": self.turn_counter
        }
        return d
    
    def get_observation(self, player_idx: int) -> Dict:
        """
        Get the observation for a specific player, suitable for ML models
        This includes only information that would be available to the player
        """
        state = self.state
        player = self.players[player_idx]
        
        observation = {
            "flipped_card": state["flipped_card"],
            "tokens_on_card": state["tokens_on_card"],
            "my_hand": sorted(player.hand),
            "my_tokens": player.tokens,
            "my_score": player.score,
            "my_position": player.player_number,
            "is_my_turn": player_idx == self.turn,
            "cards_remaining": state["cards_remaining"],
            "other_players": []
        }
        
        # Add visible information about other players
        for i, p in enumerate(self.players):
            if i != player_idx:
                observation["other_players"].append({
                    "position": p.player_number,
                    "hand": sorted(p.hand),
                    "tokens": p.tokens,
                    "score": p.score
                })
                
        return observation
    
    def get_vectorized_observation(self, player_idx: int) -> np.ndarray:
        """
        Convert the game state to a fixed-size vector representation for RL algorithms
        """
        obs = self.get_observation(player_idx)
        
        # Initialize feature vector with zeros
        # Vector includes:
        # - flipped card (one-hot encoded, 36 values for cards 0-35)
        # - tokens on card (normalized value)
        # - my hand (one-hot encoded, 36 values)
        # - my tokens (normalized value)
        # - other players' hands (one-hot encoded, n_players * 36 values)
        # - other players' tokens (normalized values, n_players values)
        
        # Adjust size based on max number of players
        max_players = 7  # Maximum supported by the game
        vector_size = 36 + 1 + 36 + 1 + (max_players-1) * 36 + (max_players-1)
        
        vec = np.zeros(vector_size)
        
        # Encode flipped card (one-hot)
        if obs["flipped_card"] > 0:
            vec[obs["flipped_card"]] = 1
            
        # Encode tokens on card (normalized by max possible tokens)
        max_possible_tokens = sum(player.tokens for player in self.players)
        vec[36] = obs["tokens_on_card"] / max_possible_tokens if max_possible_tokens > 0 else 0
        
        # Encode my hand (one-hot)
        for card in obs["my_hand"]:
            vec[37 + card] = 1
            
        # Encode my tokens (normalized)
        start_tokens = 11  # Maximum starting tokens per player
        vec[37 + 36] = obs["my_tokens"] / start_tokens
        
        # Encode other players' hands and tokens
        other_offset = 37 + 36 + 1
        for i, other in enumerate(obs["other_players"]):
            if i >= max_players - 1:
                break
                
            # Other player's hand (one-hot)
            for card in other["hand"]:
                vec[other_offset + i * 36 + card] = 1
                
            # Other player's tokens (normalized)
            vec[other_offset + (max_players-1) * 36 + i] = other["tokens"] / start_tokens
            
        return vec
    
    def get_valid_actions(self, player_idx: int = None) -> List[int]:
        """
        Get the valid actions for a player (for RL)
        Actions: 0 = No Thanks (pass), 1 = Take Card
        """
        if player_idx is None:
            player_idx = self.turn
            
        player = self.players[player_idx]
        
        # If it's not the player's turn, no actions are valid
        if player_idx != self.turn:
            return []
            
        # If player has no tokens, they can only take the card
        if player.tokens == 0:
            return [1]
            
        # Otherwise, they can either pass or take
        return [0, 1]
    
    def get_action_mask(self, player_idx: int = None) -> np.ndarray:
        """
        Returns a binary mask (0/1) indicating valid actions for RL
        """
        valid_actions = self.get_valid_actions(player_idx)
        mask = np.zeros(2)  # Binary action space: [pass, take]
        
        for action in valid_actions:
            mask[action] = 1
            
        return mask
    
    def get_text_description(self) -> str:
        """
        Returns a text description of the current game state,
        suitable for large language models
        """
        state = self.state
        current_player = self.players[self.turn]
        
        description = [
            f"Game State: Turn {self.turn_counter}, Player {self.turn}'s turn.",
            f"Current card: {state['flipped_card']} with {state['tokens_on_card']} tokens on it.",
            f"Cards remaining in deck: {state['cards_remaining']}.",
            f"\nCurrent player (Player {self.turn}):",
            f"  Hand: {sorted(current_player.hand)}",
            f"  Tokens: {current_player.tokens}",
            f"  Current score: {current_player.score}",
            f"\nOther players:"
        ]
        
        for i, player in enumerate(self.players):
            if i != self.turn:
                description.append(
                    f"  Player {i}: Hand {sorted(player.hand)}, "
                    f"Tokens: {player.tokens}, Score: {player.score}"
                )
                
        return "\n".join(description)
    
    def _record_state(self):
        """Record the current state in history"""
        self.history.append(self.state)
    
    def _calculate_rewards(self):
        """
        Calculate per-step rewards for all players
        Returns a list of rewards, one for each player
        """
        rewards = [0] * self.n_players
        
        # Get current scores
        current_scores = [player.score for player in self.players]
        
        # Calculate score changes since the last reward calculation
        for i in range(self.n_players):
            # For new rewards, compare to previous rewards_history
            # Negative change in score is good (lower score is better)
            rewards[i] = self.rewards_history[i] - current_scores[i]
            
            # Update rewards history
            self.rewards_history[i] = current_scores[i]
            
        return rewards
    
    def player_action(self, action: Optional[int] = None) -> List[float]:
        """
        Function to allow the player to take an action
        
        Parameters:
        ----------
        action: Optional[int]
            If provided, forces the current player to take this action
            0 = Say "No Thanks" (pass), 1 = Take the card
            If None, asks the player's strategy to decide
            
        Returns:
        -------
        List[float]: The rewards for each player after this action
        """
        # Get the player whose turn it is
        player = self.players[self.turn]
        rewards = [0] * self.n_players

        if self.verbose:
            time.sleep(1)
            print(f"It's Player {self.turn}'s Turn")
            print(f"  The flipped card is {self.deck.flipped_card} and it's got {self.deck.tokens} tokens")
            print(f"  Player {self.turn}'s hand is {player.hand}")

        # Determine the action to take
        if action is not None:
            choice = bool(action)
        else:
            # Ask the player what they want to do
            choice = player.decide(self.state)
            
        # The player decides to take the card,
        # or is forced to because they don't have any tokens
        if choice or (player.tokens == 0):
            if self.verbose:
                if not choice and player.tokens == 0:
                    print(f"  Player {self.turn} is out of tokens")
                print(f"  Player {self.turn} chooses to take the card")

            card, tokens = self.deck.take_card()
            player.hand.append(card)
            player.hand.sort()
            player.tokens += tokens
            
            # Record state after action
            self._record_state()
            
            # Calculate rewards
            rewards = self._calculate_rewards()
            
        # The player says "No Thanks!"
        else:
            if self.verbose:
                print(f"  Player {self.turn} says NoThanks!! and skips the card.")

            player.tokens -= 1
            self.deck.tokens += 1
            self.turn_counter += 1
            
            # Record state after action
            self._record_state()
            
            # Calculate rewards
            rewards = self._calculate_rewards()

        return rewards
    
    def play_game(self, external_actions: List[int] = None) -> Tuple[Any, List[Dict]]:
        """
        Play the game until completion
        
        Parameters:
        ----------
        external_actions: List[int]
            Optional list of actions to take, useful for ML training
            If provided, must be the same length as the game
            
        Returns:
        -------
        Tuple[Player, List[Dict]]: The winning player and game history
        """
        action_idx = 0
        total_rewards = [0] * self.n_players
        
        while self.deck.has_cards:
            if external_actions and action_idx < len(external_actions):
                action = external_actions[action_idx]
                action_idx += 1
                rewards = self.player_action(action)
            else:
                rewards = self.player_action()
                
            # Accumulate rewards
            for i in range(self.n_players):
                total_rewards[i] += rewards[i]
        
        # Final scoring
        player_scores = [(i, p.score) for i, p in enumerate(self.players)]
        player_scores.sort(key=lambda x: x[1])
        
        # Final game statistics
        game_stats = {
            "winner": player_scores[0][0],
            "final_scores": [p.score for p in self.players],
            "accumulated_rewards": total_rewards,
            "total_turns": self.turn_counter
        }
        
        # Add final game stats to history
        self.history.append(game_stats)
        
        return self.players[player_scores[0][0]], self.history
    
    def reset(self) -> Dict:
        """Reset the game to initial state (for RL)"""
        # Create a new deck
        self.deck = Deck()
        self.turn_counter = 0
        self.history = []
        
        # Reset player hands and tokens
        n_tokens = {3: 11, 4: 11, 5: 11, 6: 9, 7: 7}
        for player in self.players:
            player.hand = []
            player.tokens = n_tokens[self.n_players]
            
        # Reset rewards history
        self.rewards_history = [0] * self.n_players
        
        # Record initial state
        self._record_state()
        
        return self.state
    
    def serialize(self) -> Dict:
        """Serialize the game state to a dictionary"""
        return {
            "deck": self.deck.serialize(),
            "turn_counter": self.turn_counter,
            "n_players": self.n_players,
            "players": [
                {
                    "hand": player.hand,
                    "tokens": player.tokens,
                    "player_number": player.player_number,
                    "type": player.__class__.__name__
                }
                for player in self.players
            ],
            "history": self.history
        }
    
    @classmethod
    def deserialize(cls, data: Dict, player_classes: Dict) -> 'Game':
        """Create a game from serialized data and player class mapping"""
        game = cls([])
        game.deck = Deck.deserialize(data["deck"])
        game.turn_counter = data["turn_counter"]
        game.n_players = data["n_players"]
        game.history = data["history"]
        
        # Recreate players
        game.players = []
        for p_data in data["players"]:
            player_class = player_classes[p_data["type"]]
            player = player_class()
            player.hand = p_data["hand"]
            player.tokens = p_data["tokens"]
            player.assign_position(p_data["player_number"])
            game.players.append(player)
            
        return game