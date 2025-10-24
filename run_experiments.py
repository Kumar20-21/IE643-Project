"""
Convenience script to run common experiments
Usage examples:
    python run_experiments.py train
    python run_experiments.py evaluate
    python run_experiments.py full --episodes 500
"""

import argparse
import os
import sys
from datetime import datetime

from abides_dqn_env import (
    Config, 
    OptimalExecutionEnv, 
    DQNAgent, 
    train_dqn, 
    set_seed
)

from play_abides_dqn import (
    evaluate_agent,
    visualize_single_episode,
    compare_with_baselines,
    plot_training_curves
)


def run_training(args):
    """Run training experiment"""
    print("\n" + "="*70)
    print("TRAINING EXPERIMENT")
    print("="*70)
    
    set_seed(args.seed)
    
    # Create config
    config = Config()
    config.NUM_EPISODES = args.episodes
    
    if args.fast:
        print("\nRunning FAST mode (reduced episodes for testing)")
        config.NUM_EPISODES = 100
        config.SAVE_FREQ = 50
    
    # Train
    agent, rewards, shortfalls = train_dqn(
        config, 
        save_dir=args.save_dir,
        use_abides=args.use_abides
    )
    
    # Plot training curves
    if args.plot:
        plot_path = os.path.join(args.save_dir, 'training_curves.png')
        plot_training_curves(rewards, shortfalls, save_path=plot_path)
    
    return agent, rewards, shortfalls


def run_evaluation(args):
    """Run evaluation experiment"""
    print("\n" + "="*70)
    print("EVALUATION EXPERIMENT")
    print("="*70)
    
    set_seed(args.seed)
    
    # Load model
    if not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        print("Train a model first with: python run_experiments.py train")
        return None
    
    # Initialize
    config = Config()
    env = OptimalExecutionEnv(config, use_abides=args.use_abides, seed=args.seed)
    agent = DQNAgent(config)
    agent.load(args.model)
    
    # Evaluate
    results = evaluate_agent(
        agent, 
        env, 
        num_episodes=args.eval_episodes,
        render=args.render,
        verbose=True
    )
    
    # Save results
    results_dir = os.path.join(args.save_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'eval_results_{timestamp}.txt')
    
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Episodes: {args.eval_episodes}\n")
        f.write(f"Seed: {args.seed}\n\n")
        
        for key, value in results.items():
            if key not in ['episode_rewards', 'episode_shortfalls', 
                          'episode_completions', 'episode_trades', 'episode_avg_prices']:
                f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def run_visualization(args):
    """Run visualization experiment"""
    print("\n" + "="*70)
    print("VISUALIZATION EXPERIMENT")
    print("="*70)
    
    set_seed(args.seed)
    
    # Load model
    if not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        return None
    
    # Initialize
    config = Config()
    env = OptimalExecutionEnv(config, use_abides=args.use_abides, seed=args.seed)
    agent = DQNAgent(config)
    agent.load(args.model)
    
    # Visualize
    results_dir = os.path.join(args.save_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_path = os.path.join(results_dir, f'episode_viz_{timestamp}.png')
    
    summary = visualize_single_episode(agent, env, save_path=vis_path)
    
    return summary


def run_comparison(args):
    """Run baseline comparison experiment"""
    print("\n" + "="*70)
    print("BASELINE COMPARISON EXPERIMENT")
    print("="*70)
    
    set_seed(args.seed)
    
    # Load model
    if not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        return None
    
    # Initialize
    config = Config()
    env = OptimalExecutionEnv(config, use_abides=args.use_abides, seed=args.seed)
    agent = DQNAgent(config)
    agent.load(args.model)
    
    # Compare
    results = compare_with_baselines(agent, env, num_episodes=args.eval_episodes)
    
    # Save comparison
    results_dir = os.path.join(args.save_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_file = os.path.join(results_dir, f'comparison_{timestamp}.txt')
    
    with open(comp_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BASELINE COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Episodes per strategy: {args.eval_episodes}\n")
        f.write(f"Seed: {args.seed}\n\n")
        f.write(f"{'Strategy':<15} {'Mean Reward':<20} {'Mean Shortfall (bps)':<20}\n")
        f.write("-"*70 + "\n")
        
        for strategy, res in results.items():
            shortfall_str = f"{res['mean_shortfall_bps']:.2f}" if res['mean_shortfall_bps'] else "N/A"
            f.write(f"{strategy:<15} {res['mean_reward']:>8.2f} ± {res['std_reward']:>6.2f}   {shortfall_str:>10}\n")
    
    print(f"\nComparison saved to: {comp_file}")
    
    return results


def run_full_pipeline(args):
    """Run complete training and evaluation pipeline"""
    print("\n" + "="*70)
    print("FULL PIPELINE: TRAIN → EVALUATE → VISUALIZE → COMPARE")
    print("="*70)
    
    # Train
    print("\n>>> Step 1/4: Training...")
    agent, rewards, shortfalls = run_training(args)
    
    # Update model path to use the trained model
    args.model = os.path.join(args.save_dir, 'dqn_final.pt')
    
    # Evaluate
    print("\n>>> Step 2/4: Evaluating...")
    eval_results = run_evaluation(args)
    
    # Visualize
    print("\n>>> Step 3/4: Visualizing...")
    viz_summary = run_visualization(args)
    
    # Compare
    print("\n>>> Step 4/4: Comparing with baselines...")
    comp_results = run_comparison(args)
    
    print("\n" + "="*70)
    print("FULL PIPELINE COMPLETE!")
    print("="*70)
    print(f"All results saved to: {args.save_dir}/results/")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run ABIDES DQN experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training test (100 episodes)
  python run_experiments.py train --fast
  
  # Full training (1000 episodes)
  python run_experiments.py train --episodes 1000
  
  # Evaluate trained model
  python run_experiments.py evaluate --model models/dqn_final.pt
  
  # Visualize single episode
  python run_experiments.py visualize --model models/dqn_final.pt
  
  # Compare with baselines
  python run_experiments.py compare --model models/dqn_final.pt
  
  # Run complete pipeline
  python run_experiments.py full --episodes 500
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Experiment type')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train DQN agent')
    train_parser.add_argument('--episodes', type=int, default=1000,
                             help='Number of training episodes')
    train_parser.add_argument('--fast', action='store_true',
                             help='Fast mode (100 episodes for testing)')
    train_parser.add_argument('--plot', action='store_true', default=False,
                             help='Plot training curves')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained agent')
    eval_parser.add_argument('--model', type=str, default='models/dqn_final.pt',
                            help='Path to trained model')
    eval_parser.add_argument('--eval_episodes', type=int, default=100,
                            help='Number of evaluation episodes')
    eval_parser.add_argument('--render', action='store_true',
                            help='Render episodes')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Visualize single episode')
    viz_parser.add_argument('--model', type=str, default='models/dqn_final.pt',
                           help='Path to trained model')
    
    # Comparison command
    comp_parser = subparsers.add_parser('compare', help='Compare with baselines')
    comp_parser.add_argument('--model', type=str, default='models/dqn_final.pt',
                            help='Path to trained model')
    comp_parser.add_argument('--eval_episodes', type=int, default=100,
                            help='Number of episodes per strategy')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    full_parser.add_argument('--episodes', type=int, default=1000,
                            help='Number of training episodes')
    full_parser.add_argument('--eval_episodes', type=int, default=100,
                            help='Number of evaluation episodes')
    full_parser.add_argument('--fast', action='store_true',
                            help='Fast mode (100 training episodes)')
    
    # Common arguments for all commands
    for p in [train_parser, eval_parser, viz_parser, comp_parser, full_parser]:
        p.add_argument('--seed', type=int, default=42,
                      help='Random seed')
        p.add_argument('--save_dir', type=str, default='models',
                      help='Directory to save/load models')
        p.add_argument('--use_abides', action='store_true',
                      help='Use full ABIDES simulator (if available)')
    
    args = parser.parse_args()
    
    # Dispatch to appropriate function
    if args.command == 'train':
        run_training(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'visualize':
        run_visualization(args)
    elif args.command == 'compare':
        run_comparison(args)
    elif args.command == 'full':
        run_full_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
