import os
import json
import time
import argparse
import random
from typing import List, Dict, Any, Optional, Tuple

# Import our game modules
from game_ml import Game
from ml_players import Player, LLMPlayer, BasicMath, NetScore, AaronsRules

# Example LLM API interface (modify to match your specific LLM provider)
class LLMBridge:
    """
    Bridge class for connecting Large Language Models to the No Thanks game.
    
    This provides a standardized interface for different LLM providers
    (OpenAI, Anthropic, local models, etc.) to interact with the game.
    """
    
    def __init__(self, provider="mock", api_key=None, model_name=None, temperature=0.2):
        """
        Initialize the LLM bridge
        
        Parameters:
        ----------
        provider: str
            LLM provider ('openai', 'anthropic', 'local', 'mock')
        api_key: str
            API key for the LLM provider
        model_name: str
            Name of the model to use
        temperature: float
            Temperature parameter for generation
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        
        # Set up provider-specific configuration
        if self.provider == "openai":
            try:
                import openai
                openai.api_key = self.api_key
                self.client = openai.Client()
            except ImportError:
                print("OpenAI library not found. Install with: pip install openai")
                self.provider = "mock"
                
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("Anthropic library not found. Install with: pip install anthropic")
                self.provider = "mock"
                
        elif self.provider == "local":
            try:
                # For local models like llama.cpp
                from llama_cpp import Llama
                self.model = Llama(model_path=self.model_name)
            except ImportError:
                print("llama-cpp-python not found. Install for local models.")
                self.provider = "mock"
                
        # Tracking for interactions
        self.conversation_history = []
        self.game_states = []
        self.decisions = []
        
    def get_decision(self, game_state: Dict, verbose: bool = False) -> bool:
        """
        Get a decision from the LLM about the current game state
        
        Parameters:
        ----------
        game_state: Dict
            The current game state
        verbose: bool
            Whether to print the interaction
            
        Returns:
        -------
        bool: True to take the card, False to say "No Thanks"
        """
        # Save game state
        self.game_states.append(game_state)
        
        # Create prompt from game state
        prompt = self._create_game_prompt(game_state)
        
        # Get response from LLM
        if verbose:
            print("\n--- SENDING TO LLM ---")
            print(prompt)
            print("----------------------")
            
        response = self._query_llm(prompt)
        
        if verbose:
            print("\n--- LLM RESPONSE ---")
            print(response)
            print("--------------------")
            
        # Parse decision
        decision = self._parse_decision(response)
        
        # Save decision
        self.decisions.append({
            "prompt": prompt,
            "response": response,
            "parsed_decision": decision,
            "timestamp": time.time()
        })
        
        return decision
    
    def _create_game_prompt(self, game_state: Dict) -> str:
        """Create a prompt describing the game state for the LLM"""
        # Get player state (assuming this is for LLM player)
        llm_player_idx = game_state["player_turn_index"]
        llm_player_state = game_state["player_states"][llm_player_idx]
        
        # Create system instruction
        system_instruction = """
You are playing the card game 'No Thanks!'. Your goal is to get the lowest score possible.

Rules:
1. Each card has a point value equal to the number on the card
2. Cards in sequential runs only count the lowest card in the run (e.g. a run of 7-8-9 only counts 7 points)
3. Each token reduces your score by 1 point
4. On your turn, you can either take the current card, or place a token on it and pass
5. If you take a card, you also get all tokens on it
6. If you have no tokens, you must take the card
7. Lower score is better

You must respond with TAKE to take the card, or PASS to pass (placing a token). 
Explain your reasoning, then end with either:
DECISION: TAKE
or 
DECISION: PASS
"""
        
        # Game state description
        prompt = [
            f"Current card: {game_state['flipped_card']} with {game_state['tokens_on_card']} tokens on it",
            f"Cards remaining in deck: {game_state['cards_remaining']}",
            f"Your hand: {sorted(llm_player_state['hand'])}",
            f"Your tokens: {llm_player_state['tokens']}",
            f"Your current score: {llm_player_state['score']} (lower is better)",
            "\nOther players:"
        ]
        
        # Add information about other players
        for i, player_state in enumerate(game_state["player_states"]):
            if i != llm_player_idx:
                prompt.append(
                    f"  Player {player_state['index']}: "
                    f"Hand {sorted(player_state['hand'])}, "
                    f"Tokens: {player_state['tokens']}, "
                    f"Score: {player_state['score']}"
                )
                
        # Add forced decision if no tokens
        if llm_player_state["tokens"] == 0:
            prompt.append("\nYou have 0 tokens, so you must take the card.")
            
        # Ask for decision
        prompt.append("\nWhat is your decision? Take the card or pass?")
        
        # Configure provider-specific format
        if self.provider == "openai":
            # OpenAI API format
            return [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": "\n".join(prompt)}
            ]
        elif self.provider == "anthropic":
            # Anthropic Claude format
            return f"{system_instruction}\n\nHuman: {' '.join(prompt)}\n\nAssistant:"
        else:
            # Simple format for other providers
            return f"{system_instruction}\n\n{' '.join(prompt)}"
            
    def _query_llm(self, prompt: Any) -> str:
        """Query the LLM with the given prompt"""
        try:
            if self.provider == "openai":
                # OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    temperature=self.temperature,
                    max_tokens=300
                )
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                # Anthropic API
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=300,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.provider == "local":
                # Local model
                response = self.model(
                    prompt,
                    max_tokens=300,
                    temperature=self.temperature
                )
                return response["choices"][0]["text"]
                
            else:
                # Mock response for testing
                return self._mock_response(prompt)
                
        except Exception as e:
            print(f"Error querying LLM: {e}")
            # Return a default response in case of error
            return "I'm having trouble deciding. DECISION: TAKE"
    
    def _mock_response(self, prompt: Any) -> str:
        """Generate a mock response for testing"""
        # Extract game state from prompt to make a somewhat intelligent decision
        import re
        
        # Parse card and tokens
        card_match = re.search(r"Current card: (\d+)", str(prompt))
        tokens_match = re.search(r"with (\d+) tokens", str(prompt))
        
        if card_match and tokens_match:
            card = int(card_match.group(1))
            tokens = int(tokens_match.group(1))
            
            # Simple strategy: take if value - tokens < 15
            if card - tokens < 15:
                return f"""
I see the current card is {card} with {tokens} tokens.
Taking this card would cost me {card} points but gain me {tokens} tokens, for a net value of {card-tokens}.
Since {card-tokens} is less than 15, it's a good deal.
DECISION: TAKE
"""
            else:
                return f"""
I see the current card is {card} with {tokens} tokens.
Taking this card would cost me {card} points but gain me {tokens} tokens, for a net value of {card-tokens}.
Since {card-tokens} is more than 15, it's not a good deal yet.
DECISION: PASS
"""
        
        # Forced decision if tokens = 0
        if "You have 0 tokens" in str(prompt):
            return "I have no tokens left, so I must take the card. DECISION: TAKE"
            
        # Random decision as fallback
        if random.random() < 0.5:
            return "After analyzing the game state, I'll take the card. DECISION: TAKE"
        else:
            return "After analyzing the game state, I'll pass on this card. DECISION: PASS"
            
    def _parse_decision(self, response: str) -> bool:
        """
        Parse the LLM's response to determine the decision
        Returns: True for TAKE, False for PASS
        """
        # Look for explicit decision marker
        if "DECISION: TAKE" in response.upper():
            return True
        elif "DECISION: PASS" in response.upper():
            return False
            
        # Otherwise, check for take/pass keywords
        response_upper = response.upper()
        
        # Check for explicit TAKE/PASS statements
        if "I WILL TAKE" in response_upper or "I'LL TAKE" in response_upper:
            return True
        if "I WILL PASS" in response_upper or "I'LL PASS" in response_upper:
            return False
            
        # More lenient checks using just the words
        take_words = ["TAKE", "ACCEPT", "YES"]
        pass_words = ["PASS", "NO THANKS", "DECLINE", "NO"]
        
        # Count occurrences of decisional words
        take_count = sum(response_upper.count(word) for word in take_words)
        pass_count = sum(response_upper.count(word) for word in pass_words)
        
        # If more take words than pass words, take the card
        if take_count > pass_count:
            return True
        elif pass_count > take_count:
            return False
        
        # Default to TAKE if we can't determine
        return True
        
    def save_conversation(self, filename: str):
        """Save the conversation history to a file"""
        data = {
            "provider": self.provider,
            "model": self.model_name,
            "interactions": self.decisions,
            "game_states": self.game_states
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


class NoThanksLLMRunner:
    """
    Runner class for playing No Thanks with LLM agents
    """
    
    def __init__(self, llm_bridge, num_players=4, opponent_types=None):
        """
        Initialize the runner
        
        Parameters:
        ----------
        llm_bridge: LLMBridge
            The LLM bridge to use for LLM players
        num_players: int
            Number of players (3-7)
        opponent_types: List[Tuple[type, Tuple]]
            List of opponent types and their constructor arguments
        """
        self.llm_bridge = llm_bridge
        self.num_players = num_players
        
        # Default opponent types if none provided
        if opponent_types is None:
            self.opponent_types = [
                (BasicMath, (12,)),
                (NetScore, (12,)),
                (AaronsRules, (5, 5))
            ]
        else:
            self.opponent_types = opponent_types
            
        # Game statistics
        self.games_played = 0
        self.llm_wins = 0
        self.game_results = []
        
    def setup_game(self, llm_position=None):
        """
        Set up a new game with one LLM player and other opponents
        
        Parameters:
        ----------
        llm_position: int
            Position for the LLM player (0 to num_players-1)
            If None, a random position will be chosen
            
        Returns:
        -------
        Tuple[Game, int]: The game object and LLM player position
        """
        # Choose LLM position if not specified
        if llm_position is None:
            llm_position = random.randint(0, self.num_players - 1)
            
        # Create LLM player
        llm_player = LLMPlayer(self.llm_bridge)
        
        # Create all players
        players = []
        opponent_idx = 0
        
        for i in range(self.num_players):
            if i == llm_position:
                players.append(llm_player)
            else:
                # Select an opponent type
                opponent_class, args = self.opponent_types[opponent_idx % len(self.opponent_types)]
                players.append(opponent_class(*args))
                opponent_idx += 1
                
        # Create game
        game = Game(players=players, verbose=False)
        
        return game, llm_position
        
    def play_game(self, llm_position=None, verbose=False):
        """
        Play a single game
        
        Parameters:
        ----------
        llm_position: int
            Position for the LLM player
        verbose: bool
            Whether to print game progress
            
        Returns:
        -------
        Dict: Game results
        """
        # Set up game
        game, llm_position = self.setup_game(llm_position)
        
        if verbose:
            print(f"Starting game with LLM player at position {llm_position}")
            
        # Play the game
        winner, history = game.play_game()
        
        # Determine results
        final_scores = [player.score for player in game.players]
        llm_player = game.players[llm_position]
        llm_score = llm_player.score
        
        # Calculate rankings
        sorted_players = sorted([(i, p.score) for i, p in enumerate(game.players)], 
                               key=lambda x: x[1])
        rankings = {player_idx: rank for rank, (player_idx, _) in enumerate(sorted_players)}
        llm_rank = rankings[llm_position]
        
        # Record results
        is_llm_win = winner is llm_player
        if is_llm_win:
            self.llm_wins += 1
            
        self.games_played += 1
        
        # Create results dict
        results = {
            "game_number": self.games_played,
            "is_llm_win": is_llm_win,
            "final_scores": final_scores,
            "llm_score": llm_score,
            "llm_rank": llm_rank,
            "player_types": [player.__class__.__name__ for player in game.players],
            "llm_position": llm_position
        }
        
        self.game_results.append(results)
        
        if verbose:
            print("\nGame results:")
            print(f"Winner: {winner.__class__.__name__} (Player {game.players.index(winner)})")
            print("Final scores:")
            for i, player in enumerate(game.players):
                print(f"  Player {i} ({player.__class__.__name__}): {player.score}")
            print(f"LLM player ranked {llm_rank + 1} out of {self.num_players}")
            
        return results
        
    def run_games(self, num_games=10, verbose=False):
        """
        Run multiple games
        
        Parameters:
        ----------
        num_games: int
            Number of games to play
        verbose: bool
            Whether to print game progress
            
        Returns:
        -------
        Dict: Summary of game results
        """
        for i in range(num_games):
            if verbose:
                print(f"\nPlaying game {i+1}/{num_games}")
                
            self.play_game(verbose=verbose)
            
            if verbose and (i+1) % 10 == 0:
                win_rate = (self.llm_wins / (i+1)) * 100
                print(f"Current LLM win rate: {win_rate:.1f}%")
                
        # Calculate summary statistics
        win_rate = self.llm_wins / num_games
        
        # Get average ranking
        avg_rank = sum(result["llm_rank"] for result in self.game_results) / num_games
        
        # Get average score
        avg_score = sum(result["llm_score"] for result in self.game_results) / num_games
        
        summary = {
            "games_played": num_games,
            "llm_wins": self.llm_wins,
            "win_rate": win_rate,
            "avg_rank": avg_rank,
            "avg_score": avg_score
        }
        
        if verbose:
            print("\nFinal results:")
            print(f"Games played: {num_games}")
            print(f"LLM wins: {self.llm_wins}")
            print(f"Win rate: {win_rate:.2%}")
            print(f"Average rank: {avg_rank:.2f}")
            print(f"Average score: {avg_score:.2f}")
            
        return summary
        
    def save_results(self, filename):
        """Save game results to a file"""
        data = {
            "games_played": self.games_played,
            "llm_wins": self.llm_wins,
            "win_rate": self.llm_wins / max(1, self.games_played),
            "results": self.game_results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            

def main():
    """Main function to run LLM games"""
    parser = argparse.ArgumentParser(description='Run No Thanks with LLM players')
    parser.add_argument('--provider', type=str, default='mock', 
                       help='LLM provider (openai, anthropic, local, mock)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key for the LLM provider')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name to use')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of games to play')
    parser.add_argument('--players', type=int, default=4,
                       help='Number of players per game (3-7)')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Temperature for LLM generations')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed game information')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Save results to the specified file')
    parser.add_argument('--save-conversation', type=str, default=None,
                       help='Save LLM conversation to the specified file')
    
    args = parser.parse_args()
    
    # Get API key from environment variable if not provided
    api_key = args.api_key
    if api_key is None:
        if args.provider == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
        elif args.provider == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    # Set default model if not provided
    model = args.model
    if model is None:
        if args.provider == 'openai':
            model = 'gpt-4'
        elif args.provider == 'anthropic':
            model = 'claude-3-opus-20240229'
        elif args.provider == 'local':
            model = 'llama-7b.gguf'  # Example path, user needs to provide actual path
            
    # Create LLM bridge
    llm_bridge = LLMBridge(
        provider=args.provider,
        api_key=api_key,
        model_name=model,
        temperature=args.temperature
    )
    
    # Create runner
    runner = NoThanksLLMRunner(
        llm_bridge=llm_bridge,
        num_players=args.players
    )
    
    # Run games
    print(f"Running {args.games} games with {args.provider} LLM...")
    summary = runner.run_games(num_games=args.games, verbose=args.verbose)
    
    # Save results if requested
    if args.save_results:
        runner.save_results(args.save_results)
        print(f"Results saved to {args.save_results}")
        
    # Save conversation if requested
    if args.save_conversation:
        llm_bridge.save_conversation(args.save_conversation)
        print(f"Conversation saved to {args.save_conversation}")
        
    return summary


if __name__ == "__main__":
    main()