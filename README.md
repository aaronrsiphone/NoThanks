# No Thanks - ML Edition

A machine learning-friendly implementation of the card game "No Thanks" for training reinforcement learning agents and Large Language Models.

## Overview

This repository contains a modified version of the "No Thanks" card game specifically tailored for machine learning applications. The game has been enhanced with:

1. Standardized observation and action spaces for reinforcement learning
2. Rich state representations in both vector and text formats
3. Flexible reward functions for training
4. Interfaces for Large Language Model (LLM) agents
5. Performance tracking and evaluation tools

## Game Rules

"No Thanks" is a simple card game with the following rules:

- Cards are numbered from 3 to 35
- 9 random cards are removed from the deck at the start
- Each player starts with a set number of tokens (depending on player count)
- On your turn, you can either take the current card or place a token on it and pass
- If you take a card, you also get all tokens on it
- If you have no tokens, you must take the card
- Scoring:
  - Each card is worth its face value in points
  - Cards in sequential runs only count the lowest card
  - Each token reduces your score by 1
  - Lowest score wins

## Files

- `game_ml.py` - Core game logic with ML-friendly interfaces
- `players.py` - Player types including RL and LLM-based players
- `ml_environment.py` - RL environment and simple model implementations
- `ml_example.py` - Example training and evaluation code
- `llm_bridge.py` - Bridge for integrating LLMs with the game

## Machine Learning Features

### Reinforcement Learning

The code provides:

- Vectorized observation spaces suitable for RL algorithms
- Discrete action spaces (take/pass)
- Per-step and episodic rewards
- Methods for valid action masking
- Tools for model evaluation and comparison
- Simple Q-learning implementation
- Optional PyTorch DQN implementation

### LLM Integration

For LLM-based players:

- Text-based game state descriptions optimized for LLMs
- Structured prompts with game rules
- Decision parsing and interpretation
- Support for multiple LLM providers (OpenAI, Anthropic, local models)
- Conversation tracking and analysis tools

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/aaronrsiphone/NoThanks
cd no-thanks-ml

# Install dependencies
pip install numpy matplotlib
# Optional dependencies
pip install torch openai anthropic
```

### Basic Usage

```python
# Import the necessary modules
from game_ml import Game
from ml_players import BasicMath, NetScore, AaronsRules, RLPlayer

# Create players
players = [
    BasicMath(12),
    NetScore(12),
    AaronsRules(5, 5),
    RLPlayer()  # Random play if no model provided
]

# Create and play a game
game = Game(players=players, verbose=True)
winner, history = game.play_game()

print(f"Winner: {winner.__class__.__name__}")
print(f"Final scores: {[p.score for p in game.players]}")
```

### Training an RL Agent

```python
from ml_environment import NoThanksEnv, SimpleRLModel

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
for episode in range(5000):
    state = env.reset()
    done = False
    
    while not done:
        action_mask = env.game.get_action_mask(env.player_position)
        action = model.predict(state, action_mask)
        next_state, reward, done, _ = env.step(action)
        model.update(state, action, reward, next_state, done)
        state = next_state

# Save the trained model
model.save("no_thanks_rl_model.pkl")
```

### Using an LLM Player

```python
from llm_bridge import LLMBridge, NoThanksLLMRunner

# Create LLM bridge (uses mock responses by default)
llm_bridge = LLMBridge(provider="mock")

# Create LLM runner
runner = NoThanksLLMRunner(
    llm_bridge=llm_bridge,
    num_players=4
)

# Run games
summary = runner.run_games(num_games=10, verbose=True)
print(f"LLM win rate: {summary['win_rate']:.2%}")
```

## Command-line Tools

The repository includes several command-line tools:

```bash
# Train a Q-learning agent
python ml_example.py --train-rl --episodes 5000

# Train a DQN agent (if PyTorch is available)
python ml_example.py --train-dqn --episodes 5000

# Evaluate different player strategies
python ml_example.py --evaluate --games 1000

# Play games with an LLM agent
python llm_bridge.py --provider openai --model gpt-4 --games 10 --verbose
```

## Advanced Uses

### Custom RL Models

You can implement your own RL models by:

1. Creating a new model class
2. Implementing the `predict` and `update` methods
3. Connecting it to an RLPlayer instance

### Custom LLM Integration

To use a different LLM provider:

1. Extend the `LLMBridge` class
2. Implement the `_query_llm` method for your provider
3. Update the prompt format if needed

## License

[MIT License](LICENSE)