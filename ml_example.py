import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from typing import List, Dict, Any, Optional
from collections import deque

# Import our custom modules
from game_ml import Game
from ml_players import RLPlayer, LLMPlayer, BasicMath, NetScore, AaronsRules
from ml_environment import NoThanksEnv, SimpleRLModel, LLMInterface

# Optional: Import libraries for more advanced ML if available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Example OpenAI-like LLM interface (replace with actual API calls)
class OpenAIInterface(LLMInterface):
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        super().__init__(api_key, model_name)
        self.interaction_log = []
        
    def get_decision(self, prompt: str) -> str:
        """
        Get a decision from OpenAI API
        
        Note: This is a stub implementation. In production code,
        you would make an actual API call to OpenAI here.
        """
        # Simulated response for demo purposes
        # In real implementation, make actual API call:
        # response = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     messages=[
        #         {"role": "system", "content": "You are a strategic card game player."},
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # decision = response.choices[0].message.content
        
        # Simplified logic for demo:
        # This simulates a simple LLM strategy
        if "You have 0 tokens" in prompt:
            decision = "TAKE"  # Must take card if no tokens
        else:
            # Extract card value and tokens
            import re
            card_match = re.search(r"Current card: (\d+)", prompt)
            tokens_match = re.search(r"with (\d+) tokens", prompt)
            
            if card_match and tokens_match:
                card = int(card_match.group(1))
                tokens = int(tokens_match.group(1))
                
                # Simple strategy: take card if value - tokens < 15
                if card - tokens < 15:
                    decision = "TAKE"
                else:
                    decision = "PASS"
            else:
                decision = random.choice(["TAKE", "PASS"])
        
        # Log interaction
        self.log_interaction(prompt, decision)
        
        return decision
    
    def log_interaction(self, prompt: str, response: str, metadata: Dict = None):
        """Log interaction for analysis"""
        self.interaction_log.append({
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        })
        
        # Could also write to file or database in a real implementation


# Example PyTorch-based RL model (if torch is available)
if TORCH_AVAILABLE:
    class DQNModel(nn.Module):
        """Deep Q-Network for No Thanks game"""
        def __init__(self, input_dim, output_dim, hidden_dim=128):
            super(DQNModel, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        def forward(self, x):
            return self.network(x)
    
    class DQNAgent:
        """DQN Agent implementation for No Thanks game"""
        def __init__(self, state_dim, action_dim, 
                    learning_rate=0.001, 
                    gamma=0.99,
                    epsilon=1.0,
                    epsilon_decay=0.995,
                    epsilon_min=0.01,
                    memory_size=10000,
                    batch_size=64):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            self.batch_size = batch_size
            
            # Neural network model
            self.model = DQNModel(state_dim, action_dim)
            self.target_model = DQNModel(state_dim, action_dim)
            self.target_model.load_state_dict(self.model.state_dict())
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()
            
            # Experience replay buffer
            self.memory = deque(maxlen=memory_size)
            
            # Training metrics
            self.training_episodes = 0
            self.losses = []
            
        def act(self, state, action_mask=None):
            """Choose an action based on state"""
            if np.random.rand() < self.epsilon:
                # Random action
                if action_mask is not None:
                    valid_actions = [i for i, valid in enumerate(action_mask) if valid]
                    if valid_actions:
                        return np.random.choice(valid_actions)
                return np.random.randint(self.action_dim)
            
            # Forward pass through network
            state_tensor = torch.FloatTensor(state)
            q_values = self.model(state_tensor).detach().numpy()
            
            # Apply action mask
            if action_mask is not None:
                for i, valid in enumerate(action_mask):
                    if not valid:
                        q_values[i] = -float('inf')
                        
            return np.argmax(q_values)
        
        def remember(self, state, action, reward, next_state, done):
            """Store experience in replay memory"""
            self.memory.append((state, action, reward, next_state, done))
            
        def replay(self):
            """Train the network on a batch of experiences"""
            if len(self.memory) < self.batch_size:
                return 0
                
            # Sample batch of experiences
            batch = random.sample(self.memory, self.batch_size)
            
            # Extract components
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)
            
            # Current Q-values
            current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Next Q-values (from target network)
            with torch.no_grad():
                next_q = self.target_model(next_states).max(1)[0]
                
            # Target Q-values
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
            # Compute loss
            loss = self.loss_fn(current_q, target_q)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
            
            return loss.item()
            
        def update_target_model(self):
            """Update target model with weights from main model"""
            self.target_model.load_state_dict(self.model.state_dict())
            
        def decay_epsilon(self):
            """Decay exploration rate"""
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        def save(self, filename):
            """Save model to file"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_episodes': self.training_episodes,
                'losses': self.losses
            }, filename)
            
        def load(self, filename):
            """Load model from file"""
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_episodes = checkpoint['training_episodes']
            self.losses = checkpoint['losses']

def train_dqn_agent(episodes=5000, update_target_every=10, save_path="no_thanks_dqn.pt"):
    """Train a DQN agent on the No Thanks game"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot train DQN agent.")
        return None, None, None, None
        
    # Create environment
    env = NoThanksEnv(num_players=4)
        
    # Create environment
    env = NoThanksEnv(num_players=4)
    
    # Create DQN agent
    agent = DQNAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=10000,
        batch_size=64
    )
    
    # Create RL player with DQN agent
    class DQNPlayer(RLPlayer):
        def decide(self, game_state):
            if self.tokens == 0:
                return True
                
            # Get observation from game state
            observation = env.game.get_vectorized_observation(self.player_number)
            
            # Get action mask
            action_mask = env.game.get_action_mask(self.player_number)
            
            # Choose action using DQN
            action = agent.act(observation, action_mask)
            
            return bool(action)
    
    # Replace the RL player in the environment
    env.rl_player = DQNPlayer()
    env.players[env.player_position] = env.rl_player
    
    # Training loop
    rewards = []
    losses = []
    win_rates = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Get action mask
            action_mask = env.game.get_action_mask(env.player_position)
            
            # Choose action
            action = agent.act(state, action_mask)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train network
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                losses.append(loss)
                
            # Update state and reward
            state = next_state
            total_reward += reward
            
        # Update target network periodically
        if (episode + 1) % update_target_every == 0:
            agent.update_target_model()
            
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Track metrics
        rewards.append(total_reward)
        agent.training_episodes += 1
        
        # Evaluate performance periodically
        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards[-100:]) / min(100, len(rewards))
            
            # Evaluate win rate
            win_count = 0
            eval_episodes = 20
            for _ in range(eval_episodes):
                state = env.reset()
                done = False
                
                while not done:
                    action_mask = env.game.get_action_mask(env.player_position)
                    action = agent.act(state, action_mask=action_mask)
                    next_state, _, done, info = env.step(action)
                    state = next_state
                    
                if done and info.get("rl_player_rank", -1) == 0:  # Rank 0 = winner
                    win_count += 1
                    
            win_rate = win_count / eval_episodes
            win_rates.append(win_rate)
            
            print(f"Episode {episode+1}/{episodes}, " 
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Win Rate: {win_rate:.2%}, "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    # Save the trained model
    agent.save(save_path)
    
    # Plot training metrics
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot([i*100 for i in range(len(win_rates))], win_rates)
    plt.title('Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    plt.tight_layout()
    plt.savefig('dqn_training_metrics.png')
    
    return agent, rewards, losses, win_rates


def evaluate_strategies(num_games=1000):
    """Compare different strategies including ML-based players"""
    # Player types to evaluate
    player_classes = [
        RLPlayer,           # Using the SimpleRLModel (pre-trained)
        BasicMath,
        NetScore,
        AaronsRules,
    ]
    
    if TORCH_AVAILABLE:
        player_classes.append(DQNPlayer)  # Add DQN player if available
        
    # Player constructor arguments
    player_args = [
        (),                 # RLPlayer (will be assigned model later)
        (12,),              # BasicMath threshold
        (12,),              # NetScore threshold
        (5, 5),             # AaronsRules thresholds
    ]
    
    if TORCH_AVAILABLE:
        player_args.append(())  # DQNPlayer
        
    # Load trained models if available
    rl_model = None
    if os.path.exists("no_thanks_rl_model.pkl"):
        rl_model = SimpleRLModel.load("no_thanks_rl_model.pkl")
        
    dqn_agent = None
    if TORCH_AVAILABLE and os.path.exists("no_thanks_dqn.pt"):
        # Initialize DQN agent
        env = NoThanksEnv(num_players=4)
        dqn_agent = DQNAgent(
            state_dim=env.observation_space_shape[0],
            action_dim=env.action_space
        )
        dqn_agent.load("no_thanks_dqn.pt")
    
    # Win counters for each strategy
    win_counts = {strategy.__name__: 0 for strategy in player_classes}
    score_sums = {strategy.__name__: 0 for strategy in player_classes}
    
    # Run games
    for game_idx in range(num_games):
        # Create players
        players = []
        for i, (player_class, args) in enumerate(zip(player_classes, player_args)):
            player = player_class(*args)
            
            # Assign models to ML players
            if isinstance(player, RLPlayer) and rl_model:
                player.model = rl_model
                
            if TORCH_AVAILABLE and isinstance(player, DQNPlayer) and dqn_agent:
                # Create a wrapper to make DQN agent compatible with game interface
                player.decide = lambda game_state, p=player, a=dqn_agent, e=env: (
                    True if p.tokens == 0 else bool(a.act(
                        e.game.get_vectorized_observation(p.player_number) if hasattr(e, 'game') else np.zeros(a.state_dim),
                        None
                    ))
                )
                
            players.append(player)
            
        # Randomize player order
        random.shuffle(players)
        
        # Assign positions
        for i, player in enumerate(players):
            player.assign_position(i)
            
        # Create and play game
        game = Game(players=players, verbose=False)
        winner, _ = game.play_game()
        
        # Update win count
        winner_class = winner.__class__.__name__
        win_counts[winner_class] += 1
        
        # Update score sums
        for player in players:
            score_sums[player.__class__.__name__] += player.score
            
        # Print progress
        if (game_idx + 1) % 100 == 0:
            print(f"Completed {game_idx + 1}/{num_games} games")
    
    # Calculate win rates and average scores
    win_rates = {name: count / num_games for name, count in win_counts.items()}
    avg_scores = {name: score / num_games for name, score in score_sums.items()}
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"{'Strategy':<15} {'Win Rate':<10} {'Avg Score':<10}")
    print("-" * 40)
    
    for name in win_rates:
        print(f"{name:<15} {win_rates[name]:.2%}      {avg_scores[name]:.2f}")
        
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(win_rates.keys(), win_rates.values())
    plt.title('Win Rates')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(avg_scores.keys(), avg_scores.values())
    plt.title('Average Scores (Lower is Better)')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    
    return win_rates, avg_scores


def main():
    """Main function to run various demonstrations"""
    parser = argparse.ArgumentParser(description='No Thanks ML Examples')
    parser.add_argument('--train-rl', action='store_true', help='Train the Q-learning agent')
    parser.add_argument('--train-dqn', action='store_true', help='Train the DQN agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate different strategies')
    parser.add_argument('--play-llm', action='store_true', help='Play games with an LLM agent')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--games', type=int, default=1000, help='Number of evaluation games')
    
    args = parser.parse_args()
    
    # Train Q-learning agent
    if args.train_rl:
        print("Training Q-learning agent...")
        model, rewards = train_rl_agent(episodes=args.episodes)
        
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title('RL Agent Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('rl_training_progress.png')
        plt.close()
        
    # Train DQN agent
    if args.train_dqn and TORCH_AVAILABLE:
        print("Training DQN agent...")
        train_dqn_agent(episodes=args.episodes)
        
    # Evaluate strategies
    if args.evaluate:
        print(f"Evaluating strategies over {args.games} games...")
        evaluate_strategies(num_games=args.games)
        
    # Play with LLM
    if args.play_llm:
        print("Playing games with LLM agent...")
        llm_interface = OpenAIInterface()
        play_with_llm(llm_interface, num_games=10)
        
    if not any([args.train_rl, args.train_dqn, args.evaluate, args.play_llm]):
        print("No action specified. Use --help to see available options.")


if __name__ == "__main__":
    main()