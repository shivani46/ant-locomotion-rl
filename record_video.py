import os
import glob
import subprocess
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordVideo

BEST_MODEL_DIR = "best_model"
MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model")
VEC_NORMALIZE_PATH = os.path.join(BEST_MODEL_DIR, "vec_normalize.pkl")
VIDEO_DIR = "videos"
OUTPUT_PATH = os.path.join(VIDEO_DIR, "ant_walking.mp4")
N_EPISODES = 3


def make_env():
    base_env = gym.make("Ant-v5", render_mode="rgb_array")
    base_env = RecordVideo(
        base_env,
        video_folder=VIDEO_DIR,
        name_prefix="ant_ep",
        episode_trigger=lambda ep: True,
    )
    return base_env


def main():
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"No model found at '{MODEL_PATH}.zip'. Run train.py first."
        )

    os.makedirs(VIDEO_DIR, exist_ok=True)

    env = DummyVecEnv([make_env])

    if os.path.exists(VEC_NORMALIZE_PATH):
        env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize statistics.")

    model = PPO.load(MODEL_PATH, env=env)
    print(f"Loaded model from {MODEL_PATH}")

    for episode in range(1, N_EPISODES + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            steps += 1
        print(f"Episode {episode}: reward={total_reward:.2f}, steps={steps}")

    env.close()

    clips = sorted(glob.glob(os.path.join(VIDEO_DIR, "ant_ep*.mp4")))

    if not clips:
        print("No clips recorded — check that render_mode='rgb_array' is supported.")
        return

    if len(clips) == 1:
        os.rename(clips[0], OUTPUT_PATH)
    else:
        list_file = os.path.join(VIDEO_DIR, "_concat.txt")
        with open(list_file, "w") as f:
            for c in clips:
                f.write(f"file '{os.path.abspath(c)}'\n")
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_file, "-c", "copy", OUTPUT_PATH],
            capture_output=True, text=True,
        )
        os.remove(list_file)
        for c in clips:
            os.remove(c)
        if result.returncode != 0:
            print(f"ffmpeg merge failed:\n{result.stderr}")
            return

    print(f"Saved {N_EPISODES}-episode video to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
