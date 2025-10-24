"""
Play with trained ABIDES DQN agent
Evaluate, visualize, and test the trained optimal execution agent

Usage:
    python play_with_abides_dqn.py --model models/dqn_final.pt --episodes 100
    python play_with_abides_dqn.py --model models/dqn_final.pt --visualize
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple
import os
import argparse
from datetime import datetime

# Import from abides_dqn_env.py
from abides_dqn_env import (
    Config, 
    OptimalExecutionEnv, 
    DQNAgent, 
    set_seed
)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_agent(agent: DQNAgent, env: OptimalExecutionEnv, 
                   num_episodes: int = 100, render: bool = False, 
                   verbose: bool = True) -> Dict:
    """
    Evaluate trained agent over multiple episodes.
    
    Args:
        agent: Trained DQN agent
        env: Environment
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        verbose: Print detailed info
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_shortfalls = []
    episode_completions = []
    episode_trades = []
    episode_steps = []
    episode_avg_prices = []
    
    if verbose:
        print("\n" + "="*70)
        print(f"Evaluating agent over {num_episodes} episodes...")
        print("="*70 + "\n")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Select action (greedy - no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            step_count += 1
            
            if render and episode == 0:  # Render first episode
                print(f"Step {step_count}: Action={action}, Order={info['order_size']:.0f}, "
                      f"Remaining={info['remaining_inventory']}")
        
        # Get execution summary
        summary = env.get_execution_summary()
        
        episode_rewards.append(episode_reward)
        episode_completions.append(1 if summary['completed'] else 0)
        episode_trades.append(summary['num_trades'])
        episode_steps.append(step_count)
        
        if summary['completed'] and summary['total_executed'] > 0:
            episode_shortfalls.append(summary['implementation_shortfall_bps'])
            episode_avg_prices.append(summary['avg_execution_price'])
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Completed={summary['completed']}, "
                  f"Trades={summary['num_trades']}")
    
    # Calculate statistics
    results = {
        'num_episodes': num_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_shortfall_bps': np.mean(episode_shortfalls) if episode_shortfalls else None,
        'std_shortfall_bps': np.std(episode_shortfalls) if episode_shortfalls else None,
        'completion_rate': np.mean(episode_completions) * 100,
        'mean_trades': np.mean(episode_trades),
        'mean_steps': np.mean(episode_steps),
        'episode_rewards': episode_rewards,
        'episode_shortfalls': episode_shortfalls,
        'episode_completions': episode_completions,
        'episode_trades': episode_trades,
        'episode_avg_prices': episode_avg_prices
    }
    
    if verbose:
        print("\n" + "="*70)
        print("Evaluation Results:")
        print("="*70)
        print(f"Episodes: {results['num_episodes']}")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        if results['mean_shortfall_bps'] is not None:
            print(f"Mean Implementation Shortfall: {results['mean_shortfall_bps']:.2f} ± {results['std_shortfall_bps']:.2f} bps")
        print(f"Completion Rate: {results['completion_rate']:.1f}%")
        print(f"Mean Trades per Episode: {results['mean_trades']:.1f}")
        print(f"Mean Steps per Episode: {results['mean_steps']:.1f}")
        print("="*70 + "\n")
    
    return results


def visualize_single_episode(agent: DQNAgent, env: OptimalExecutionEnv, 
                             save_path: str = None):
    """
    Run and visualize a single episode in detail.
    
    Args:
        agent: Trained DQN agent
        env: Environment
        save_path: Path to save figure
    """
    state = env.reset()
    
    # Track trajectory
    steps = []
    actions = []
    rewards = []
    prices = []
    inventories = []
    order_sizes = []
    
    done = False
    step_count = 0
    
    print("\n" + "="*70)
    print("Running single episode with visualization...")
    print("="*70 + "\n")
    
    while not done:
        action = agent.select_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        steps.append(step_count)
        actions.append(action)
        rewards.append(reward)
        prices.append(info['avg_execution_price'] if info['order_size'] > 0 else env.arrival_price)
        inventories.append(info['remaining_inventory'])
        order_sizes.append(info['order_size'])
        
        state = next_state
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Step {step_count}: Inventory={info['remaining_inventory']}, "
                  f"Cumulative Reward={sum(rewards):.2f}")
    
    summary = env.get_execution_summary()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Inventory over time
    axes[0, 0].plot(steps, inventories, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Remaining Inventory (shares)')
    axes[0, 0].set_title('Inventory Execution Profile')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Order sizes
    axes[0, 1].bar(steps, order_sizes, width=1.0, alpha=0.7)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Order Size (shares)')
    axes[0, 1].set_title('Order Sizes Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cumulative reward
    cumulative_rewards = np.cumsum(rewards)
    axes[1, 0].plot(steps, cumulative_rewards, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Cumulative Reward')
    axes[1, 0].set_title('Cumulative Reward Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Action distribution
    action_counts = [actions.count(i) for i in range(env.action_space_n)]
    axes[1, 1].bar(range(env.action_space_n), action_counts, alpha=0.7)
    axes[1, 1].set_xlabel('Action')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Action Distribution')
    axes[1, 1].set_xticks(range(env.action_space_n))
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Episode Analysis\n'
                 f'Total Reward: {sum(rewards):.2f}, '
                 f'Shortfall: {summary.get("implementation_shortfall_bps", 0):.2f} bps, '
                 f'Completed: {summary["completed"]}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
    else:
        plt.show()
    
    return summary


def compare_with_baselines(agent: DQNAgent, env: OptimalExecutionEnv, 
                           num_episodes: int = 100):
    """
    Compare DQN agent with baseline strategies.
    
    Baselines:
    1. TWAP (Time-Weighted Average Price) - uniform execution
    2. Aggressive - execute everything immediately
    3. Passive - execute at the end
    
    Args:
        agent: Trained DQN agent
        env: Environment
        num_episodes: Number of episodes for comparison
    """
    print("\n" + "="*70)
    print("Comparing DQN with Baseline Strategies")
    print("="*70 + "\n")
    
    results = {}
    
    # 1. DQN Agent
    print("1. Evaluating DQN Agent...")
    results['DQN'] = evaluate_agent(agent, env, num_episodes, verbose=False)
    
    # 2. TWAP Strategy
    print("2. Evaluating TWAP Baseline...")
    twap_rewards = []
    twap_shortfalls = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Uniform execution
        orders_per_step = env.total_size / env.max_steps
        
        while not done:
            # Execute uniform amount (approximately action 1 consistently)
            action = 1  # Q_min shares
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        twap_rewards.append(episode_reward)
        summary = env.get_execution_summary()
        if summary['completed']:
            twap_shortfalls.append(summary['implementation_shortfall_bps'])
    
    results['TWAP'] = {
        'mean_reward': np.mean(twap_rewards),
        'std_reward': np.std(twap_rewards),
        'mean_shortfall_bps': np.mean(twap_shortfalls) if twap_shortfalls else None,
    }
    
    # 3. Aggressive Strategy
    print("3. Evaluating Aggressive Baseline...")
    aggressive_rewards = []
    aggressive_shortfalls = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Always use maximum action
            action = env.action_space_n - 1  # Maximum order size
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        aggressive_rewards.append(episode_reward)
        summary = env.get_execution_summary()
        if summary['completed']:
            aggressive_shortfalls.append(summary['implementation_shortfall_bps'])
    
    results['Aggressive'] = {
        'mean_reward': np.mean(aggressive_rewards),
        'std_reward': np.std(aggressive_rewards),
        'mean_shortfall_bps': np.mean(aggressive_shortfalls) if aggressive_shortfalls else None,
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("Comparison Results:")
    print("="*70)
    print(f"{'Strategy':<15} {'Mean Reward':<20} {'Mean Shortfall (bps)':<20}")
    print("-"*70)
    
    for strategy, res in results.items():
        shortfall_str = f"{res['mean_shortfall_bps']:.2f}" if res['mean_shortfall_bps'] else "N/A"
        print(f"{strategy:<15} {res['mean_reward']:>8.2f} ± {res['std_reward']:>6.2f}   {shortfall_str:>10}")
    
    print("="*70 + "\n")
    
    return results


def plot_training_curves(rewards: List[float], shortfalls: List[float], 
                         save_path: str = None):
    """
    Plot training curves.
    
    Args:
        rewards: Episode rewards
        shortfalls: Episode shortfalls
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    axes[0].plot(rewards, alpha=0.3, label='Raw')
    
    # Moving average
    window = 50
    if len(rewards) >= window:
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        axes[0].plot(moving_avg, linewidth=2, label=f'{window}-episode MA')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot shortfalls
    if shortfalls:
        axes[1].plot(shortfalls, alpha=0.3, label='Raw')
        
        if len(shortfalls) >= window:
            moving_avg = pd.Series(shortfalls).rolling(window=window).mean()
            axes[1].plot(moving_avg, linewidth=2, label=f'{window}-episode MA')
        
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Implementation Shortfall (bps)')
        axes[1].set_title('Implementation Shortfall Over Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize trained DQN agent')
    parser.add_argument('--model', type=str, default='models/dqn_final.pt',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize a single episode')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with baseline strategies')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set seed
    set_seed(args.seed)
    
    print("\n" + "="*70)
    print("ABIDES DQN Optimal Execution - Evaluation")
    print("="*70 + "\n")
    
    # Initialize
    config = Config()
    env = OptimalExecutionEnv(config, use_abides=False, seed=args.seed)
    agent = DQNAgent(config)
    
    # Load trained model
    if os.path.exists(args.model):
        agent.load(args.model)
        print(f"✓ Model loaded from {args.model}\n")
    else:
        print(f"Error: Model file not found: {args.model}")
        print("Please train a model first using Abides_dqn.py")
        return
    
    # Evaluate
    if not args.visualize and not args.compare:
        results = evaluate_agent(agent, env, args.episodes, render=False)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.save_dir, f'eval_results_{timestamp}.txt')
        with open(results_file, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("="*50 + "\n")
            for key, value in results.items():
                if key not in ['episode_rewards', 'episode_shortfalls', 'episode_completions', 
                              'episode_trades', 'episode_avg_prices']:
                    f.write(f"{key}: {value}\n")
        print(f"Results saved to {results_file}")
    
    # Visualize single episode
    if args.visualize:
        vis_path = os.path.join(args.save_dir, 'episode_visualization.png')
        visualize_single_episode(agent, env, save_path=vis_path)
    
    # Compare with baselines
    if args.compare:
        compare_with_baselines(agent, env, num_episodes=args.episodes)


if __name__ == "__main__":
    main()
