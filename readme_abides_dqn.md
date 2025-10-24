# ABIDES Optimal Execution with DQN

Implementation of the paper: **"Optimal Execution with Reinforcement Learning"** (arXiv:2411.06389)

This implementation provides an end-to-end DQN agent for optimal order execution using the ABIDES market simulator.

## 📁 Project Structure

```
your_project/
├── ABIDES/                    # ABIDES repository (https://github.com/abides-sim/abides)
│   ├── Kernel.py
│   ├── agent/
│   │   ├── ExchangeAgent.py
│   │   ├── TradingAgent.py
│   │   └── ...
│   ├── util/
│   └── ...
├── Abides_dqn.py             # Main implementation file
├── play_with_abides_dqn.py   # Evaluation and visualization
├── models/                    # Saved models (created automatically)
└── results/                   # Evaluation results (created automatically)
```

## 🔧 Installation

### 1. Clone ABIDES Repository

```bash
# Clone ABIDES into your project directory
git clone https://github.com/abides-sim/abides.git ABIDES
```

### 2. Install Dependencies

```bash
pip install torch numpy pandas matplotlib
```

### 3. Install ABIDES Dependencies

Follow the ABIDES installation instructions from their repository.

## 📊 Paper Specifications Implemented

### MDP Formulation

**State (9-dimensional):**
- Percentage holdings remaining
- Percentage time remaining
- Volume imbalance at 5 LOB levels (5 values)
- Best bid price
- Best ask price

**Action Space (5 discrete actions):**
- Action 0: Do nothing
- Actions 1-4: Execute Q_min × k shares (k=1,2,3,4)
  - Q_min = 20 shares

**Reward Function (Paper's Equation 5):**
```
r_t = Q_t^k · (P_0 - P_t) - α·d_t
```
Where:
- Q_t^k: Quantity executed at time t
- P_0: Arrival price (benchmark)
- P_t: Average execution price
- d_t: Depth consumed by the order
- α = 2: Depth penalty coefficient

**Terminal Penalties:**
- 5 per non-executed share
- 5 per over-executed share

**Discount Factor:** γ = 0.9999

### Environment Parameters

- **Total order size:** 20,000 shares
- **Time window:** 30 minutes (1800 seconds)
- **Control frequency:** 1 second per step → 1800 max steps
- **State history:** Stack last 4 states for stability

### DQN Architecture

**Network:**
- Input: State (9 × 4 = 36 dimensions with history)
- Hidden layers: [50, 20] neurons
- Output: Q-values for 5 actions

**Training Hyperparameters:**
- Learning rate: 1e-3 → 0 (linear decay over 90,000 steps)
- Epsilon: 1.0 → 0.02 (decay over 10,000 steps)
- Batch size: 64
- Replay buffer: 100,000 transitions
- Target network update: Every 1,000 steps
- Gradient clipping: 10.0

## 🚀 Usage

### Quick Start

```bash
# Run training with environment test
python Abides_dqn.py
```

The script will:
1. Test the environment
2. Ask if you want to start training
3. Train for 1000 episodes
4. Save models every 100 episodes

### Training Only

```python
from Abides_dqn import train_dqn, Config, set_seed

# Set seed for reproducibility
set_seed(42)

# Create configuration
config = Config()

# Train agent
agent, rewards, shortfalls = train_dqn(config, save_dir="models", use_abides=False)
```

### Custom Configuration

```python
from Abides_dqn import Config, train_dqn

# Customize configuration
config = Config()
config.NUM_EPISODES = 2000
config.TOTAL_SIZE = 10000  # Smaller order
config.LEARNING_RATE_START = 5e-4
config.BATCH_SIZE = 128

# Train with custom config
agent, rewards, shortfalls = train_dqn(config)
```

## 📈 Evaluation and Visualization

### Basic Evaluation

```bash
# Evaluate trained agent over 100 episodes
python play_with_abides_dqn.py --model models/dqn_final.pt --episodes 100
```

### Visualize Single Episode

```bash
# Visualize execution trajectory
python play_with_abides_dqn.py --model models/dqn_final.pt --visualize
```

This creates a 4-panel visualization showing:
1. Inventory execution profile
2. Order sizes over time
3. Cumulative reward
4. Action distribution

### Compare with Baselines

```bash
# Compare DQN with TWAP and Aggressive strategies
python play_with_abides_dqn.py --model models/dqn_final.pt --compare --episodes 100
```

### Programmatic Evaluation

```python
from Abides_dqn import Config, OptimalExecutionEnv, DQNAgent
from play_with_abides_dqn import evaluate_agent, visualize_single_episode, compare_with_baselines

# Load trained agent
config = Config()
env = OptimalExecutionEnv(config, use_abides=False, seed=42)
agent = DQNAgent(config)
agent.load("models/dqn_final.pt")

# Evaluate
results = evaluate_agent(agent, env, num_episodes=100)

# Visualize
summary = visualize_single_episode(agent, env, save_path="results/episode.png")

# Compare with baselines
comparison = compare_with_baselines(agent, env, num_episodes=100)
```

## 🔄 ABIDES Integration

Currently, the implementation uses a **simplified LOB simulation** for standalone testing. To integrate with full ABIDES:

### Step 1: Enable ABIDES

```python
from Abides_dqn import Config, OptimalExecutionEnv

config = Config()
env = OptimalExecutionEnv(config, use_abides=True)  # Enable ABIDES
```

### Step 2: Implement ABIDES Connection

The `OptimalExecutionEnv` class has placeholders for ABIDES integration:

```python
def _setup_abides_kernel(self):
    """Setup ABIDES kernel with exchange and background agents"""
    # Create kernel
    kernelStartTime = pd.Timestamp('2025-01-01 09:30:00')
    kernelStopTime = pd.Timestamp('2025-01-01 16:00:00')
    
    self.kernel = Kernel(
        "Optimal Execution Simulation",
        random_state=np.random.RandomState(seed=self.seed)
    )
    
    # Add Exchange, Value agents, Noise agents, etc.
    # See paper's specifications for OU process with jumps
```

### ABIDES Market Configuration (from paper)

```python
# Fundamental value process (OU with jumps)
R_BAR = 100000  # Fundamental value (cents)
KAPPA = 1.94e-15  # Mean reversion rate
SIGMA_S = 0  # Volatility of fundamental
MEGASHOCK_LAMBDA = 2.77778e-13  # Jump intensity
MEGASHOCK_MEAN = 1e3  # Jump mean
MEGASHOCK_VAR = 5e4  # Jump variance

# Background agents
NUM_VALUE_AGENTS = 100
NUM_MOMENTUM_AGENTS = 25
NUM_NOISE_AGENTS = 5000
```

## 📊 Expected Results

Based on the paper's experiments, you should observe:

### Training Progress
- **Initial episodes:** Random exploration, negative rewards
- **Mid training (episodes 100-500):** Learning optimal pacing
- **Late training (episodes 500+):** Convergence to near-optimal policy

### Performance Metrics
- **Completion rate:** Should approach 100%
- **Implementation shortfall:** Lower than TWAP baseline
- **Reward:** Increasing trend over training

### Learned Behavior
The agent should learn to:
1. **Pace execution** throughout the time window
2. **Adapt to market conditions** (volume imbalance)
3. **Balance** execution urgency vs. market impact
4. **Avoid excessive depth consumption**

## 🛠️ Customization

### Modify State Space

```python
def _get_state(self) -> np.ndarray:
    # Add your custom features
    custom_feature = compute_custom_feature()
    
    state = np.array([
        pct_holdings,
        pct_time,
        *volume_imbalance,
        best_bid,
        best_ask,
        custom_feature  # Your addition
    ], dtype=np.float32)
    
    return state
```

### Modify Reward Function

```python
def _calculate_reward(self, order_size, avg_price, depth_consumed):
    # Custom reward shaping
    base_reward = (self.arrival_price - avg_price) * order_size
    depth_penalty = self.config.DEPTH_PENALTY_ALPHA * depth_consumed
    
    # Add custom components
    urgency_bonus = compute_urgency_bonus()
    
    reward = base_reward - depth_penalty + urgency_bonus
    return reward
```

### Change Network Architecture

```python
config = Config()
config.HIDDEN_LAYERS = [128, 64, 32]  # Deeper network
```

## 📝 Output Files

### Training
- `models/dqn_episode_100.pt` - Model checkpoints every 100 episodes
- `models/dqn_final.pt` - Final trained model

### Evaluation
- `results/eval_results_YYYYMMDD_HHMMSS.txt` - Evaluation metrics
- `results/episode_visualization.png` - Episode trajectory plots

### Model Checkpoint Format

```python
{
    'q_network': state_dict,
    'target_network': state_dict,
    'optimizer': state_dict,
    'steps_done': int,
    'episodes_done': int
}
```

## 🐛 Troubleshooting

### ABIDES Import Error

**Problem:** `Warning: Could not import ABIDES components`

**Solution:** 
1. Ensure ABIDES folder is in your project directory
2. Check ABIDES path in code: `abides_path = os.path.join(os.getcwd(), 'ABIDES')`
3. Falls back to simplified LOB simulation automatically

### CUDA Out of Memory

**Problem:** GPU memory error during training

**Solution:**
```python
config = Config()
config.BATCH_SIZE = 32  # Reduce batch size
config.MEMORY_SIZE = 50000  # Reduce replay buffer
```

### Training Not Converging

**Problem:** Reward not improving

**Solutions:**
1. **Increase exploration:** 
   ```python
   config.EPSILON_DECAY_STEPS = 20000  # Explore longer
   ```

2. **Adjust learning rate:**
   ```python
   config.LEARNING_RATE_START = 5e-4  # Lower LR
   ```

3. **Check reward scaling:**
   - Rewards should be roughly in [-1000, 1000] range
   - Terminal penalties may dominate; adjust if needed

### Incomplete Executions

**Problem:** Agent not executing full order

**Solution:**
```python
config.PENALTY_NON_EXECUTED = 10  # Increase penalty
```

## 📚 Paper Reference

```bibtex
@article{optimal_execution_rl_2024,
  title={Optimal Execution with Reinforcement Learning},
  author={[Authors]},
  journal={arXiv preprint arXiv:2411.06389},
  year={2024}
}
```

## 🎯 Key Implementation Details

### State History Stacking
- Concatenates last 4 states for temporal information
- Helps DQN learn execution velocity and trends
- Final state dimension: 9 × 4 = 36

### Epsilon-Greedy Exploration
- Starts at 100% random actions
- Decays to 2% over 10,000 steps
- Ensures sufficient exploration early

### Learning Rate Schedule
- Linear decay from 1e-3 to 0
- Over 90,000 training steps
- Helps fine-tune policy late in training

### Target Network
- Updated every 1,000 steps
- Stabilizes training by providing consistent targets
- Critical for DQN convergence

### Gradient Clipping
- Clips gradients to [-10, 10]
- Prevents exploding gradients
- Important in noisy market environment

## 💡 Tips for Best Results

1. **Start with simplified LOB** before enabling full ABIDES
2. **Monitor completion rate** - should be >95% after training
3. **Compare with baselines** to validate learning
4. **Try multiple random seeds** - paper evaluates over multiple seeds
5. **Visualize episodes** to understand learned behavior
6. **Adjust penalties** if agent develops unwanted behavior

## 🔗 Resources

- **Paper:** https://arxiv.org/abs/2411.06389
- **ABIDES Repository:** https://github.com/abides-sim/abides
- **DQN Paper:** Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)

## 📧 Support

For issues related to:
- **This implementation:** Check troubleshooting section above
- **ABIDES simulator:** Refer to ABIDES documentation
- **Paper details:** See original paper on arXiv

---

**Happy Training! 🚀**