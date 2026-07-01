import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

TOTAL_TIMESTEPS = 3_000_000
N_ENVS = 4
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


def make_env():
    return gym.make("Ant-v5")


def main():
    train_env = make_vec_env(make_env, n_envs=N_ENVS)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.02,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_ant",
        save_vecnormalize=True,
    )

    print("Starting PPO training on Ant-v5 for 1M timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(os.path.join(BEST_MODEL_DIR, "final_model"))
    train_env.save(os.path.join(BEST_MODEL_DIR, "vec_normalize.pkl"))
    print("Training complete. Model saved to best_model/")


if __name__ == "__main__":
    main()
