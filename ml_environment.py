import numpy as np
import random
import json
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import deque
import os
import pickle

# Import the game and player classes
from game_ml import Game
from ml_players import RLPlayer, LLMPlayer, Player

class NoThanksEnv:
    """
    Reinforcement Learning Environment for the No Thanks card game.
    Follows a gym-like interface for compatibility with RL frameworks.
    """
    
    def __init__(self, num_players=4, opponent_classes=None, opponent_args=None, player_position=None, seed=None):
        """
        Initialize the environment
        
        Parameters:
        ----------
        num_players: int
            Number of players in the game (3-7)
        opponent_classes: List[type]
            List of player classes for opponents
        opponent_args: List[Tuple]
            Arguments for opponent classes initialization
        player_position: int
            Position of the RL player (0 to num_players-1)
        seed: int
            Random seed for reproducibility
        """
        self.num_players = num_players
        
        # Set random seed
        if seed is not None:
            self.seed(seed)
            
        # Default opponent setup if not provided
        if opponent_classes is None:
            from ml_players import BasicMath, NetScore, AaronsRules
            opponent_classes = [BasicMath, NetScore, AaronsRules]
            
        if opponent_args is None:
            opponent_args = [(12,), (12,), (5, 5)]
            
        # Default player position
        if player_position is None:
            self.player_position = 0
        else:
            self.player_position = player_position
            
        # Set up opponents and players
        self.setup_players(opponent_classes, opponent_args)
        
        # Initialize game
        self.game = None
        self.reset()
        
        # Action and observation spaces
        self.action_space = 2  # Binary: 0 = Pass, 1 = Take
        self.observation_space_shape = (self.game.get_vectorized_observation(self.player_position).shape[0],)
        
        # Tracking metrics
        self.reward_history = []
        self.episode_rewards = []
        
    def seed(self, seed: int = None) -> List[int]:
        """Set random seed for reproducibility"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]
    
    def setup_players(self, opponent_classes, opponent_args):
        """Set up players for the game"""
        # Create the RL player
        self.rl_player = RLPlayer()
        
        # Create opponents
        self.players = []
        opponent_idx = 0
        
        for i in range(self.num_players):
            if i == self.player_position:
                # RL player position
                self.players.append(self.rl_player)
            else:
                # Add an opponent
                opponent_class = opponent_classes[opponent_idx % len(opponent_classes)]
                opponent_arg = opponent_args[opponent_idx % len(opponent_args)]
                self.players.append(opponent_class(*opponent_arg))
                opponent_idx += 1
                
    def reset(self) -> np.ndarray:
        """Reset the environment to start a new game"""
        # Create new game
        self.game = Game(players=self.players, verbose=False)
        
        # Reset episode tracking
        self.current_episode_reward = 0
        
        # Return initial observation
        return self.get_observation()
    
    def get_observation(self) -> np.ndarray:
        """Get the current observation for the RL player"""
        return self.game.get_vectorized_observation(self.player_position)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment
        
        Parameters:
        ----------
        action: int
            Action to take (0 = Pass, 1 = Take)
            
        Returns:
        -------
        Tuple containing:
            - observation: np.ndarray
            - reward: float
            - done: bool
            - info: Dict with additional information
        """
        # Check if it's the RL player's turn
        if self.game.turn != self.player_position:
            # Fast-forward until it's the RL player's turn
            while self.game.turn != self.player_position and self.game.deck.has_cards:
                self.game.player_action()  # Other players take their turns
                
        # If game ended while fast-forwarding
        if not self.game.deck.has_cards:
            # Game is over
            final_scores = [p.score for p in self.game.players]
            rl_player_score = final_scores[self.player_position]
            
            # Calculate reward based on rank (better rank = higher reward)
            sorted_scores = sorted([(i, s) for i, s in enumerate(final_scores)], key=lambda x: x[1])
            ranks = {player_idx: rank for rank, (player_idx, _) in enumerate(sorted_scores)}
            rl_rank = ranks[self.player_position]
            
            # Normalize reward based on rank (1 for winning, decreasing for worse ranks)
            reward = (self.num_players - rl_rank) / self.num_players
            
            # Additional reward based on gap to best score
            best_score = sorted_scores[0][1]
            score_gap = max(0, (rl_player_score - best_score) / 35)  # Normalize by max card value
            
            # Final reward (rank-based + score gap penalty)
            reward = reward - score_gap
            
            self.current_episode_reward += reward
            self.episode_rewards.append(self.current_episode_reward)
            
            return self.get_observation(), reward, True, {
                "final_scores": final_scores,
                "rl_player_score": rl_player_score,
                "rl_player_rank": rl_rank,
                "episode_reward": self.current_episode_reward
            }
            
        # Check if action is valid
        valid_actions = self.game.get_valid_actions(self.player_position)
        if action not in valid_actions:
            # Invalid action, use a valid one instead
            if valid_actions:
                action = valid_actions[0]
            else:
                # No valid actions (shouldn't happen)
                raise ValueError("No valid actions available")
                
        # Take action in the game
        rewards = self.game.player_action(action)
        
        # Extract the reward for the RL player
        reward = rewards[self.player_position]
        self.current_episode_reward += reward
        
        # Fast-forward until it's the RL player's turn again
        while self.game.turn != self.player_position and self.game.deck.has_cards:
            self.game.player_action()  # Other players take their turns
                
        # Check if game has ended
        done = not self.game.deck.has_cards
        
        if done:
            # Game is over
            final_scores = [p.score for p in self.game.players]
            rl_player_score = final_scores[self.player_position]
            
            # Calculate reward based on rank (better rank = higher reward)
            sorted_scores = sorted([(i, s) for i, s in enumerate(final_scores)], key=lambda x: x[1])
            ranks = {player_idx: rank for rank, (player_idx, _) in enumerate(sorted_scores)}
            rl_rank = ranks[self.player_position]
            
            # Final reward boost based on rank
            rank_reward = (self.num_players - rl_rank) / self.num_players
            reward += rank_reward
            
            self.current_episode_reward += rank_reward
            self.episode_rewards.append(self.current_episode_reward)
            
            info = {
                "final_scores": final_scores,
                "rl_player_score": rl_player_score,
                "rl_player_rank": rl_rank,
                "episode_reward": self.current_episode_reward
            }
        else:
            info = {
                "current_scores": [p.score for p in self.game.players],
                "rl_player_score": self.game.players[self.player_position].score,
                "current_card": self.game.deck.flipped_card,
                "tokens_on_card": self.game.deck.tokens
            }
            
        return self.get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """Render the current game state"""
        state = self.game.state
        current_player = self.game.players[self.game.turn]
        
        print("-" * 50)
        print(f"Game State: Turn {self.game.turn_counter}, Player {self.game.turn}'s turn")
        print(f"Current card: {state['flipped_card']} with {state['tokens_on_card']} tokens")
        print(f"Cards remaining: {len(self.game.deck.deck)}")
        
        print("\nPlayers:")
        for i, player in enumerate(self.game.players):
            print(f"  Player {i}{' (RL)' if i == self.player_position else ''}: "
                  f"Hand {sorted(player.hand)}, Tokens: {player.tokens}, "
                  f"Score: {player.score}")
            
        print("-" * 50)
        
    def get_game_state_for_llm(self) -> str:
        """Get a text description of the game state for LLM-based agents"""
        return self.game.get_text_description()
    
    def close(self):
        """Clean up resources"""
        pass


class LLMInterface:
    """
    Interface for connecting Large Language Models to the game.
    This is a base class that should be extended with specific LLM implementations.
    """
    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key
        self.model_name = model_name
        
    def get_decision(self, prompt: str) -> str:
        """
        Get a decision from the LLM based on the provided prompt
        
        Parameters:
        ----------
        prompt: str
            Text description of the game state and decision to be made
            
        Returns:
        -------
        str: The LLM's response
        """
        # Base implementation returns a random decision
        return random.choice(["TAKE", "PASS"])
    
    def log_interaction(self, prompt: str, response: str, metadata: Dict = None):
        """Log an interaction with the LLM for analysis"""
        pass


class SimpleRLModel:
    """
    A simple RL model implementation for testing
    Implements Q-learning for the No Thanks game
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int = 2, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.99,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01):
        """
        Initialize the RL model
        
        Parameters:
        ----------
        input_dim: int
            Dimension of the input observation
        output_dim: int
            Number of possible actions
        learning_rate: float
            Learning rate for Q-learning
        discount_factor: float
            Discount factor for future rewards
        exploration_rate: float
            Initial exploration rate
        exploration_decay: float
            Rate at which exploration decays
        min_exploration_rate: float
            Minimum exploration rate
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table with a simple discretization approach
        self.q_table = {}
        
        # For tracking learning progress
        self.training_episodes = 0
        self.rewards_history = []
        
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize a continuous state for Q-table lookup
        This is a simple binning approach
        """
        # For each dimension, discretize into bins
        # This is a simple approach - more sophisticated discretization may be needed
        bins = 10  # Number of bins per dimension
        
        # Clip values to [0, 1] range and discretize
        discrete_state = tuple((np.clip(state, 0, 1) * (bins - 1)).astype(int))
        
        return discrete_state
        
    def predict(self, state: np.ndarray, action_mask: List[int] = None) -> int:
        """
        Choose an action based on the current state
        
        Parameters:
        ----------
        state: np.ndarray
            The current state observation
        action_mask: List[int]
            Binary mask of valid actions
            
        Returns:
        -------
        int: The chosen action
        """
        # Apply exploration strategy
        if random.random() < self.exploration_rate:
            # Random action, respecting action mask
            if action_mask:
                valid_actions = [i for i, valid in enumerate(action_mask) if valid]
                if valid_actions:
                    return random.choice(valid_actions)
            return random.randint(0, self.output_dim - 1)
        
        # Discretize state for Q-table lookup
        discrete_state = self._discretize_state(state)
        
        # If state not in Q-table, add it with zeros
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.output_dim)
            
        # Choose action with highest Q-value (respecting action mask)
        q_values = self.q_table[discrete_state]
        
        if action_mask:
            # Apply action mask - set invalid actions to very negative values
            masked_q_values = q_values.copy()
            for i, valid in enumerate(action_mask):
                if not valid:
                    masked_q_values[i] = -float('inf')
            return np.argmax(masked_q_values)
        
        return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: Optional[np.ndarray] = None, done: bool = False):
        """
        Update the Q-values based on the observed reward
        
        Parameters:
        ----------
        state: np.ndarray
            The state before action
        action: int
            The action taken
        reward: float
            The reward received
        next_state: Optional[np.ndarray]
            The resulting state after action
        done: bool
            Whether the episode is done
        """
        # Discretize states
        discrete_state = self._discretize_state(state)
        
        # If state not in Q-table, add it
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.output_dim)
            
        # Get current Q-value
        current_q = self.q_table[discrete_state][action]
        
        # Calculate new Q-value
        if next_state is not None and not done:
            discrete_next_state = self._discretize_state(next_state)
            
            # If next state not in Q-table, add it
            if discrete_next_state not in self.q_table:
                self.q_table[discrete_next_state] = np.zeros(self.output_dim)
                
            # Update with Q-learning formula
            max_next_q = np.max(self.q_table[discrete_next_state])
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
        else:
            # Terminal state or no next_state provided
            new_q = current_q + self.learning_rate * (reward - current_q)
            
        # Update Q-table
        self.q_table[discrete_state][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
    def train_episode(self, env, max_steps=1000):
        """Train for one episode"""
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            # Get action
            action_mask = env.game.get_action_mask(env.player_position)
            action = self.predict(state, action_mask)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-values
            self.update(state, action, reward, next_state, done)
            
            # Update tracking variables
            state = next_state
            total_reward += reward
            steps += 1
            
        self.training_episodes += 1
        self.rewards_history.append(total_reward)
        
        return total_reward, steps
    
    def save(self, filename: str):
        """Save the model to a file"""
        data = {
            "q_table": self.q_table,
            "exploration_rate": self.exploration_rate,
            "training_episodes": self.training_episodes,
            "rewards_history": self.rewards_history,
            "params": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_decay": self.exploration_decay,
                "min_exploration_rate": self.min_exploration_rate
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            
    @classmethod
    def load(cls, filename: str) -> 'SimpleRLModel':
        """Load a model from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        # Create model with same parameters
        model = cls(
            input_dim=data["params"]["input_dim"],
            output_dim=data["params"]["output_dim"],
            learning_rate=data["params"]["learning_rate"],
            discount_factor=data["params"]["discount_factor"],
            exploration_decay=data["params"]["exploration_decay"],
            min_exploration_rate=data["params"]["min_exploration_rate"]
        )
        
        # Load saved state
        model.q_table = data["q_table"]
        model.exploration_rate = data["exploration_rate"]
        model.training_episodes = data["training_episodes"]
        model.rewards_history = data["rewards_history"]
        
        return model


# Example usage

def train_rl_agent(episodes=10000, save_path="no_thanks_rl_model.pkl"):
    """Train an RL agent on the No Thanks game"""
    # Create environment
    env = NoThanksEnv(num_players=4)
    
    # Create and initialize model
    model = SimpleRLModel(
        input_dim=env.observation_space_shape[0],
        output_dim=env.action_space
    )
    
    # Connect model to player
    env.rl_player.model = model
    
    # Training loop
    episode_rewards = []
    for episode in range(episodes):
        reward, steps = model.train_episode(env)
        episode_rewards.append(reward)
        
        # Print progress periodically
        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.4f}, "
                  f"Exploration: {model.exploration_rate:.4f}")
            
    # Save the trained model
    model.save(save_path)
    
    return model, episode_rewards


def play_with_llm(llm_interface, num_games=10):
    """Play games using an LLM-based player"""
    # Create LLM player
    llm_player = LLMPlayer(llm_interface)
    
    # Create other players
    from ml_players import BasicMath, NetScore, AaronsRules
    other_players = [BasicMath(12), NetScore(12), AaronsRules(5, 5)]
    
    # Track wins
    wins = 0
    
    for game_idx in range(num_games):
        # Randomize player order
        players = [llm_player] + other_players
        random.shuffle(players)
        
        # Create and play game
        game = Game(players=players, verbose=False)
        winner, _ = game.play_game()
        
        # Check if LLM player won
        if winner is llm_player:
            wins += 1
            
        print(f"Game {game_idx+1}: Winner is {winner.__class__.__name__}")
        for p in game.players:
            print(f"  {p.__class__.__name__}: Score {p.score}")
            
    win_rate = wins / num_games
    print(f"\nLLM player won {wins}/{num_games} games (Win rate: {win_rate:.2%})")
    
    return win_rate