import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

TOTAL_TIMESTEPS = 3_000_000
N_ENVS = 4
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"

RESUME_MODEL = os.path.join(BEST_MODEL_DIR, "best_model")
RESUME_VECNORM = os.path.join(BEST_MODEL_DIR, "vec_normalize.pkl")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


def make_env():
    return gym.make("Ant-v5")


def main():
    resuming = os.path.exists(RESUME_MODEL + ".zip") and os.path.exists(RESUME_VECNORM)

    train_env = make_vec_env(make_env, n_envs=N_ENVS)
    if resuming:
        train_env = VecNormalize.load(RESUME_VECNORM, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_env, n_envs=1)
    if resuming:
        eval_env = VecNormalize.load(RESUME_VECNORM, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    if resuming:
        print(f"Resuming from {RESUME_MODEL}.zip ...")
        model = PPO.load(RESUME_MODEL, env=train_env, verbose=1)
        model.set_logger(configure(LOG_DIR, ["stdout", "tensorboard"]))
    else:
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

    mode = "resuming" if resuming else "fresh"
    print(f"Training PPO on Ant-v5 for {TOTAL_TIMESTEPS:,} timesteps ({mode})...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=not resuming,
    )

    model.save(os.path.join(BEST_MODEL_DIR, "final_model"))
    train_env.save(RESUME_VECNORM)
    print("Training complete. Model saved to best_model/")


if __name__ == "__main__":
    main()
