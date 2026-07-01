# Ant Locomotion RL — Sim-to-Real Portfolio Project

A sim-to-real reinforcement learning project training a MuJoCo Ant agent to walk using **PPO** (Proximal Policy Optimization) via Stable-Baselines3. The goal is to produce a locomotion controller in simulation that transfers to physical hardware with minimal domain adaptation.

## Environment

- **Task**: `Ant-v5` (Gymnasium + MuJoCo)
- **Observation**: 27-dim proprioceptive state (joint angles, velocities, contact forces)
- **Action**: 8-dim continuous joint torques
- **Reward**: Forward velocity + survival bonus − control cost − contact cost

## Algorithm

**PPO** with the following hyperparameters:

| Hyperparameter | Value |
|---|---|
| Learning rate | 3e-4 |
| Steps per rollout | 2048 |
| Batch size | 64 |
| Epochs per update | 10 |
| Discount (γ) | 0.99 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Parallel envs | 4 |
| Total timesteps | 1,000,000 |

Observations and rewards are normalized online with **VecNormalize**.

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd ant-locomotion-rl

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: MuJoCo 2.x is bundled with `gymnasium[mujoco]` — no separate installation needed.

## Training

```bash
python train.py
```

Outputs are saved to:
- `best_model/best_model.zip` — highest-reward checkpoint (EvalCallback)
- `best_model/final_model.zip` — model at end of training
- `best_model/vec_normalize.pkl` — normalization statistics
- `checkpoints/` — periodic checkpoints every 50k steps
- `logs/` — TensorBoard event files

### Monitor training with TensorBoard

```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` in your browser.

## Evaluation

```bash
python evaluate.py
```

Loads `best_model/best_model.zip` and `best_model/vec_normalize.pkl`, opens a MuJoCo viewer, and runs 5 episodes with deterministic actions.

## Sim-to-Real Transfer Notes

This project is structured to support sim-to-real transfer:

- **VecNormalize** keeps observation statistics that can be exported and applied on hardware.
- **Proprioceptive-only observations** (no privileged simulator state) mirror what real IMUs and encoders provide.
- **Domain randomization** can be layered in via `gymnasium` wrappers on joint friction, mass, and motor noise to improve policy robustness before hardware deployment.
- The saved normalization stats (`vec_normalize.pkl`) should be applied identically on the real robot inference loop.

## Project Structure

```
ant-locomotion-rl/
├── train.py          # PPO training with callbacks and VecNormalize
├── evaluate.py       # Render trained policy in MuJoCo viewer
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── best_model/       # Saved models and normalization stats (created at runtime)
├── checkpoints/      # Periodic checkpoints (created at runtime)
└── logs/             # TensorBoard logs (created at runtime)
```
