from itertools import groupby
from operator import itemgetter
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json

class Player:
    """
    Base class for players in the game No Thanks.
    Enhanced to be compatible with ML training.
    """

    def __init__(self):
        # set up empty hand
        self.hand = []
        self._tokens = 0  # Initialize tokens attribute
        
    def assign_position(self, player_number):
        self.player_number = player_number

    def __str__(self):
        s = f"player: {self.player_number}"
        s += f"  type: {self.__class__.__name__}"
        s += f"  tokens: {self.tokens}"
        s += f"  score: {self.score}"
        s += " hand: "
        for seq in self.sequences:
            s += f" {seq}"

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
            "index": self.player_number,
            "hand": sorted(self.hand.copy()),
            "tokens": self.tokens,
            "score": self.score,
        }
        return d
    
    def decide(self, game_state: Dict) -> bool:
        """
        Function to decide if the player should take the card

        Return true to take the card
        Return false to say "no thanks!"
        """
        n = random.randint(-1, 1)
        if n < 0:
            return False
        return True

    def test_card(self, card: int) -> int:
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
    
    def serialize(self) -> Dict:
        """Serialize the player to a dictionary"""
        return {
            "class": self.__class__.__name__,
            "hand": self.hand.copy(),
            "tokens": self.tokens,
            "player_number": self.player_number
        }

class Human(Player):
    """
    This Player allows a human to play against the computers
    """
    def decide(self, game_state):
        print(f"Its Your turn Player {self.player_number}")
        print(f"You have {self.tokens} tokens")
        
        for ps in game_state['player_states'][self.player_number+1:] + game_state['player_states'][:self.player_number]:
            print(f"Player {ps['index']} hand is {sorted(ps['hand'])}")

        choice = input("Do you want to take the card? (yes/no): ")
        if choice.lower() in ["yes", "y", "true", "1"]:
            return True
        elif choice.lower() in ["no", "n", "false", "0"]:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
            return self.decide(game_state)

class RLPlayer(Player):
    """
    A player that uses a reinforcement learning model to make decisions
    """
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.last_state = None
        self.last_action = None
        
    def decide(self, game_state):
        """
        Use the RL model to make a decision
        If no model is provided, use random actions
        """
        if self.model is None:
            # Random action if no model provided
            if self.tokens == 0:
                return True
            return random.choice([True, False])
        
        # Convert game state to model input format
        # This will depend on your specific RL model
        observation = self._preprocess_state(game_state)
        
        # Get action mask (valid actions)
        action_mask = [1, 1]  # [pass, take]
        if self.tokens == 0:
            action_mask[0] = 0  # Can't pass if no tokens
            
        # Get action from model
        action = self.model.predict(observation, action_mask)
        
        # Save for learning
        self.last_state = observation
        self.last_action = action
        
        return bool(action)
    
    def _preprocess_state(self, game_state):
        """Convert game state to model input format"""
        # This implementation depends on your specific RL model
        # For now, we'll return a simple vector
        
        # Extract relevant information
        my_state = None
        for player_state in game_state['player_states']:
            if player_state['index'] == self.player_number:
                my_state = player_state
                break
                
        if my_state is None:
            raise ValueError(f"Player {self.player_number} not found in game state")
        
        # Create feature vector
        features = [
            game_state['flipped_card'] / 35.0,  # Normalize card value
            game_state['tokens_on_card'] / 30.0,  # Normalize tokens
            len(my_state['hand']) / 24.0,  # Normalize hand size (max cards = 24)
            my_state['tokens'] / 20.0,  # Normalize tokens
        ]
        
        # Add information about sequences
        test_hand = my_state['hand'].copy()
        test_hand.append(game_state['flipped_card'])
        test_hand.sort()
        
        sequences = []
        for k, g in groupby(enumerate(test_hand), lambda i_x: i_x[0] - i_x[1]):
            sequences.append(sorted(list(map(itemgetter(1), g))))
        
        # Add sequence information
        if sequences:
            max_seq_len = max(len(seq) for seq in sequences)
            features.append(max_seq_len / 24.0)
            
            # Would taking the card create a new sequence?
            would_create_new_seq = len(sequences) > len(self.sequences)
            features.append(float(would_create_new_seq))
        else:
            features.extend([0.0, 0.0])
            
        return np.array(features)
    
    def learn(self, reward):
        """Update the model based on the reward received"""
        if self.model is not None and self.last_state is not None and self.last_action is not None:
            self.model.update(self.last_state, self.last_action, reward)

class LLMPlayer(Player):
    """
    A player that uses a Large Language Model to make decisions
    """
    def __init__(self, llm_interface=None):
        super().__init__()
        self.llm_interface = llm_interface
        self.decision_history = []
        
    def decide(self, game_state):
        """
        Use the LLM to make a decision based on the game state
        If no LLM interface is provided, use random actions
        """
        if self.llm_interface is None:
            # Random action if no LLM provided
            if self.tokens == 0:
                return True
            return random.choice([True, False])
        
        # Create a text description of the game state for the LLM
        prompt = self._create_llm_prompt(game_state)
        
        # Get decision from LLM
        llm_response = self.llm_interface.get_decision(prompt)
        
        # Parse the LLM's decision (implementation depends on your LLM interface)
        decision = self._parse_llm_response(llm_response)
        
        # Save decision history
        self.decision_history.append({
            "game_state": game_state,
            "prompt": prompt,
            "llm_response": llm_response,
            "decision": decision
        })
        
        if self.tokens == 0:
            return True
            
        return decision
    
    def _create_llm_prompt(self, game_state):
        """Create a text prompt for the LLM based on the game state"""
        my_state = None
        for player_state in game_state['player_states']:
            if player_state['index'] == self.player_number:
                my_state = player_state
                break
                
        prompt = [
            "You are playing the card game 'No Thanks!'. Here's the current game state:",
            f"Current card: {game_state['flipped_card']} with {game_state['tokens_on_card']} tokens on it.",
            f"Your hand: {sorted(my_state['hand'])}",
            f"Your tokens: {my_state['tokens']}",
            f"Your current score: {my_state['score']} (lower is better)",
            "\nOther players:",
        ]
        
        for player_state in game_state['player_states']:
            if player_state['index'] != self.player_number:
                prompt.append(
                    f"  Player {player_state['index']}: "
                    f"Hand {sorted(player_state['hand'])}, "
                    f"Tokens: {player_state['tokens']}, "
                    f"Score: {player_state['score']}"
                )
                
        prompt.extend([
            "\nIn 'No Thanks!', the rules are:",
            "1. On your turn, you can either take the current card or say 'No Thanks!' and pass.",
            "2. If you pass, you must place one token on the card.",
            "3. If you take the card, you get the card and all tokens on it.",
            "4. At the end, each card counts as points equal to its face value.",
            "5. Each token reduces your score by 1 point.",
            "6. Cards in sequential runs only count the lowest card in the run.",
            "7. Lower score is better.",
            
            "\nDecide: Should you take the card (respond with TAKE) or pass (respond with PASS)?"
        ])
        
        if my_state['tokens'] == 0:
            prompt.append("\nNote: You have 0 tokens, so you must take the card.")
            
        return "\n".join(prompt)
    
    def _parse_llm_response(self, response):
        """Parse the LLM's response to determine the decision"""
        # Simple parsing logic - can be made more robust
        response = response.upper()
        
        if "TAKE" in response:
            return True
        elif "PASS" in response:
            return False
        else:
            # Default to random if LLM response is unclear
            return random.choice([True, False])

# Original player strategies with type hints for better ML compatibility
class Denier(Player):
    """
    This player will always say "No Thanks" if able
    """
    def decide(self, _):
        if self.tokens == 0:
            return True
        return False

class BasicMath(Player):
    """
    This player inputs a threshold that is used to make a basic calculation.
    """
    def __init__(self, threshold: int = 12):
        super().__init__()
        self.threshold = threshold

    def decide(self, state: Dict) -> bool:
        if self.tokens == 0:
            return True
        if state['flipped_card'] - state['tokens_on_card'] < self.threshold:
            return True
        return False

class NetScore(BasicMath):
    """
    This player calculates what the net score impact would be of taking the card.
    """
    def decide(self, state: Dict) -> bool:
        if self.tokens == 0:
            return True
        
        card = state['flipped_card']
        tokens = state['tokens_on_card']
        score_delta = self.test_card(card) - tokens

        if score_delta < self.threshold:
            return True
        return False

class AaronsRules(Player):
    """
    This player only accepts cards that complement its current hand.
    """
    def __init__(self, 
                auto_take_threshold: int = 5,
                general_threshold: int = 5):
        super().__init__()
        self.auto_take_threshold = auto_take_threshold
        self.general_threshold = general_threshold
        
    def will_make_sequence(self, c: int, hand: List[int] = None) -> bool:
        if not hand:
            hand = self.hand
        one = (c - 1) in hand
        two = (c + 1) in hand
        return one or two

    def decide(self, state: Dict) -> bool:
        # Return True is to take.
        # Return False is Nothanks
        
        # if you dont have any tokens, take it
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
    def __init__(self, card_threshold: int = 25, tokens_threshold: int = 8):
        super().__init__()
        self.card_threshold = card_threshold
        self.tokens_threshold = tokens_threshold

    def will_make_sequence(self, c: int, hand: List[int] = None) -> bool:
        if not hand:
            hand = self.hand
        one = (c - 1) in hand
        two = (c + 1) in hand
        return one or two

    def decide(self, state: Dict) -> bool:
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