# Quick Start Guide

Get up and running with ABIDES DQN in 5 minutes!

## 🚀 Installation (2 minutes)

```bash
# 1. Clone ABIDES repository
git clone https://github.com/abides-sim/abides.git ABIDES

# 2. Install Python dependencies
pip install torch numpy pandas matplotlib

# 3. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
```

## 📦 Project Files

You should have these 4 files:
- `Abides_dqn.py` - Main implementation
- `play_with_abides_dqn.py` - Evaluation tools
- `run_experiments.py` - Convenience script
- `README.md` / `QUICKSTART.md` - Documentation

## ⚡ Quick Test (1 minute)

```bash
# Test environment and short training run
python run_experiments.py train --fast
```

This will:
- ✓ Test the environment
- ✓ Train for 100 episodes (~2-3 minutes)
- ✓ Save model to `models/dqn_final.pt`
- ✓ Generate training curves

## 🎯 Full Training (30-60 minutes)

```bash
# Train for 1000 episodes (paper specification)
python run_experiments.py train --episodes 1000
```

**Expected output:**
```
Episode  100/1000 | R:  -45234.21 | IS: 125.34 bps | Complete: 85.0% | ε: 0.900
Episode  200/1000 | R:  -32145.67 | IS:  98.21 bps | Complete: 92.0% | ε: 0.800
...
Episode 1000/1000 | R:  -18234.45 | IS:  45.67 bps | Complete: 99.0% | ε: 0.020
```

## 📊 Evaluate Results

```bash
# Evaluate trained model
python run_experiments.py evaluate --model models/dqn_final.pt --eval_episodes 100

# Visualize single episode
python run_experiments.py visualize --model models/dqn_final.pt

# Compare with baselines (TWAP, Aggressive)
python run_experiments.py compare --model models/dqn_final.pt
```

## 🔄 Complete Pipeline

```bash
# Run everything: Train → Evaluate → Visualize → Compare
python run_experiments.py full --episodes 500
```

## 📝 Alternative: Python API

```python
from Abides_dqn import Config, train_dqn, set_seed

# Set seed
set_seed(42)

# Configure
config = Config()
config.NUM_EPISODES = 500

# Train
agent, rewards, shortfalls = train_dqn(config)

# Use trained agent
from play_with_abides_dqn import evaluate_agent
from Abides_dqn import OptimalExecutionEnv

env = OptimalExecutionEnv(config)
results = evaluate_agent(agent, env, num_episodes=100)
```

## 📈 Understanding Results

### Training Metrics

**Total Reward:**
- Higher is better
- Should increase over training
- Target: -10,000 to -20,000 by end of training

**Implementation Shortfall (bps):**
- Lower is better
- Measures execution cost vs. arrival price
- Target: < 50 bps (paper results)

**Completion Rate:**
- Should reach 95-100%
- Indicates agent learned to execute full order

### What to Look For

✅ **Good Training:**
```
Episode  900 | Reward: -18234 | Shortfall: 42.3 bps | Complete: 99%
Episode 1000 | Reward: -17891 | Shortfall: 41.8 bps | Complete: 100%
```

❌ **Poor Training:**
```
Episode  900 | Reward: -89234 | Shortfall: 234.5 bps | Complete: 45%
Episode 1000 | Reward: -91234 | Shortfall: 245.2 bps | Complete: 43%
```

## 🛠️ Common Issues

### Issue: Training too slow
```python
# Reduce episodes for faster testing
python run_experiments.py train --episodes 200
```

### Issue: CUDA out of memory
```python
# In Abides_dqn.py, modify Config:
config.BATCH_SIZE = 32  # Reduce from 64
config.MEMORY_SIZE = 50000  # Reduce from 100000
```

### Issue: Agent not completing orders
```python
# In Abides_dqn.py, modify Config:
config.PENALTY_NON_EXECUTED = 10  # Increase from 5
```

### Issue: ABIDES import warning
This is **normal**! The code falls back to simplified LOB simulation automatically.
You can safely ignore this warning during initial testing.

## 📁 Output Files

After training, you'll have:

```
models/
├── dqn_episode_100.pt    # Checkpoint at episode 100
├── dqn_episode_200.pt    # Checkpoint at episode 200
├── ...
├── dqn_final.pt          # Final trained model
└── results/
    ├── eval_results_*.txt
    ├── episode_viz_*.png
    ├── comparison_*.txt
    └── training_curves.png
```

## 🎓 Next Steps

1. **Understand the code:**
   - Read `Abides_dqn.py` - See MDP formulation
   - Check `OptimalExecutionEnv` class - State/action/reward

2. **Experiment:**
   - Try different hyperparameters
   - Modify reward function
   - Change network architecture
   - Test different order sizes

3. **Integrate ABIDES:**
   - Study ABIDES documentation
   - Implement full market simulation
   - Add background agents (Value, Momentum, Noise)
   - Use OU process with jumps (as per paper)

4. **Advanced:**
   - Multiple seeds evaluation
   - Hyperparameter tuning
   - Different market conditions
   - Compare with more baselines

## 💡 Pro Tips

### Tip 1: Monitor Training
Watch the completion rate - it's the most important early indicator:
- Episode 0-100: Expect 20-60%
- Episode 100-500: Should reach 80-95%
- Episode 500+: Should be 95-100%

### Tip 2: Save Checkpoints
Models are saved every 100 episodes. If training crashes, you can resume:
```python
agent.load('models/dqn_episode_500.pt')
# Continue training from episode 500
```

### Tip 3: Visualize Early
Visualize episodes at different stages:
```bash
# Early training (random behavior)
python run_experiments.py visualize --model models/dqn_episode_100.pt

# Mid training (learning)
python run_experiments.py visualize --model models/dqn_episode_500.pt

# Final (converged)
python run_experiments.py visualize --model models/dqn_final.pt
```

### Tip 4: Compare Baselines
Always compare with baselines to validate learning:
- DQN should beat TWAP by 10-30%
- DQN should beat Aggressive by 20-50%

### Tip 5: Multiple Seeds
Run with different seeds to ensure robustness:
```bash
python run_experiments.py train --seed 42
python run_experiments.py train --seed 123
python run_experiments.py train --seed 999
```

## 🔬 Paper Reproduction Checklist

To reproduce the paper's results:

- [ ] Environment matches paper specification
  - [ ] State: 9-dim (% holdings, % time, vol_imb×5, bid, ask)
  - [ ] Action: 5 discrete (0=nothing, 1-4=k×Q_min)
  - [ ] Reward: Q·(P₀-P) - α·d
  - [ ] Penalties: 5 per non-executed share
  
- [ ] DQN architecture matches paper
  - [ ] Hidden layers: [50, 20]
  - [ ] State history: 4 frames
  - [ ] Learning rate: 1e-3 → 0 (90k steps)
  - [ ] Epsilon: 1.0 → 0.02 (10k steps)
  - [ ] Discount: γ=0.9999
  
- [ ] Training setup matches paper
  - [ ] 1000 episodes
  - [ ] Batch size: 64
  - [ ] Replay buffer: 100k
  - [ ] Target update: every 1000 steps
  - [ ] Gradient clipping: 10.0
  
- [ ] Environment parameters match paper
  - [ ] Total size: 20,000 shares
  - [ ] Time window: 30 minutes (1800s)
  - [ ] Q_min: 20 shares
  - [ ] α (depth penalty): 2

## 📊 Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 5 min | Install dependencies, verify files |
| Quick test | 3 min | Run `--fast` mode (100 episodes) |
| Full training | 30-60 min | Run 1000 episodes |
| Evaluation | 5 min | Evaluate and visualize |
| Analysis | 10 min | Compare baselines, analyze results |
| **Total** | **~1 hour** | Complete pipeline |

## 🎯 Success Criteria

You've successfully reproduced the paper if:

✅ **Completion rate** ≥ 95%
✅ **Implementation shortfall** < 60 bps (on average)
✅ **DQN beats TWAP** by ≥ 10% in reward
✅ **DQN beats Aggressive** by ≥ 20% in reward
✅ **Training converges** (reward increases, variance decreases)
✅ **Learned policy makes sense** (paces execution over time)

## 🐍 Code Snippets

### Custom Training Loop

```python
from Abides_dqn import Config, OptimalExecutionEnv, DQNAgent, set_seed

set_seed(42)
config = Config()

# Customize
config.NUM_EPISODES = 500
config.BATCH_SIZE = 128
config.LEARNING_RATE_START = 5e-4

# Initialize
env = OptimalExecutionEnv(config)
agent = DQNAgent(config)

# Training loop
for episode in range(config.NUM_EPISODES):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, next_state, reward, done)
        agent.train_step()
        
        episode_reward += reward
        state = next_state
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
        agent.save(f"models/custom_ep{episode+1}.pt")
```

### Custom Evaluation

```python
from Abides_dqn import Config, OptimalExecutionEnv, DQNAgent

config = Config()
env = OptimalExecutionEnv(config)
agent = DQNAgent(config)
agent.load("models/dqn_final.pt")

# Evaluate
rewards = []
shortfalls = []

for _ in range(100):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state
    
    rewards.append(episode_reward)
    summary = env.get_execution_summary()
    if summary['completed']:
        shortfalls.append(summary['implementation_shortfall_bps'])

print(f"Avg Reward: {np.mean(rewards):.2f}")
print(f"Avg Shortfall: {np.mean(shortfalls):.2f} bps")
```

### Modify Config on the Fly

```python
from Abides_dqn import Config

config = Config()

# Change order size
config.TOTAL_SIZE = 10000  # Smaller order

# Change time window
config.TIME_WINDOW = 900  # 15 minutes
config.MAX_STEPS = config.TIME_WINDOW // config.CONTROL_FREQUENCY

# Change penalties
config.PENALTY_NON_EXECUTED = 10
config.DEPTH_PENALTY_ALPHA = 3

# Change network
config.HIDDEN_LAYERS = [128, 64, 32]

# Change training
config.NUM_EPISODES = 2000
config.LEARNING_RATE_START = 5e-4
config.EPSILON_DECAY_STEPS = 20000
```

## 📚 Further Reading

### Papers
- **Original paper:** arXiv:2411.06389
- **DQN:** Mnih et al., "Playing Atari with Deep RL", 2013
- **ABIDES:** Byrd et al., "ABIDES: Agent-Based Interactive Discrete Event Simulator"

### Concepts
- **Optimal execution:** Minimize implementation shortfall
- **Market impact:** Temporary vs. permanent impact
- **VWAP/TWAP:** Common execution benchmarks
- **Limit order book:** Order-driven markets

### Extensions
- **Double DQN:** Van Hasselt et al., 2015
- **Dueling DQN:** Wang et al., 2016
- **Rainbow DQN:** Hessel et al., 2017
- **PPO for execution:** Proximal Policy Optimization

## 🆘 Getting Help

### Debug Mode

Add debug prints to understand behavior:

```python
# In Abides_dqn.py, modify step():
def step(self, action):
    print(f"[DEBUG] Step {self.current_step}: Action={action}, Inventory={self.remaining_inventory}")
    # ... rest of code
```

### Tensorboard Integration

For advanced monitoring (optional):

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/dqn_execution')

# In training loop:
writer.add_scalar('Reward/episode', episode_reward, episode)
writer.add_scalar('Shortfall/episode', shortfall, episode)
writer.add_scalar('Epsilon', agent._get_epsilon(), agent.steps_done)
```

### Log Everything

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Episode {episode}: Reward={reward}")
```

## ✅ Final Checklist

Before running full experiments:

- [ ] ABIDES folder is in project directory (or using simplified LOB)
- [ ] All dependencies installed (`torch`, `numpy`, `pandas`, `matplotlib`)
- [ ] Can run `python run_experiments.py train --fast` successfully
- [ ] Models directory created (done automatically)
- [ ] Have ~1 hour for full training
- [ ] Have disk space for model checkpoints (~50MB each)

## 🎉 You're Ready!

Start with:
```bash
python run_experiments.py train --fast
```

Then proceed to:
```bash
python run_experiments.py full --episodes 1000
```

Good luck! 🚀

---

**Questions?** Review the full README.md for detailed documentation.