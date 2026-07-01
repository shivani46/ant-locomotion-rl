import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

BEST_MODEL_DIR = "best_model"
MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model")
VEC_NORMALIZE_PATH = os.path.join(BEST_MODEL_DIR, "vec_normalize.pkl")
N_EVAL_EPISODES = 5


def main():
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}.zip'. Run train.py first."
        )

    env = DummyVecEnv([lambda: gym.make("Ant-v5", render_mode="human")])

    if os.path.exists(VEC_NORMALIZE_PATH):
        env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize statistics.")
    else:
        print("Warning: vec_normalize.pkl not found, running without normalization.")

    model = PPO.load(MODEL_PATH, env=env)
    print(f"Loaded model from {MODEL_PATH}")

    for episode in range(1, N_EVAL_EPISODES + 1):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1

        print(f"Episode {episode}: reward={episode_reward:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    main()
