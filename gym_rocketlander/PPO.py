import gym
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


LEARNING_RATE = 0.0003
N_STEPS = 100000
BATCH_SIZE = 256
GAMMA = 0.95
NUM_EVAL_EPISODES = 20
ENTROPY = 0.01


class DataCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.curr_reward = 0.0
        self.episode = 1
        self.mean_rewards = []

    def _on_step(self):
        step_reward = self.locals["rewards"][0].item()
        self.curr_reward += step_reward
        if self.locals["dones"][0]:
            if self.episode % 10 == 0:
                print("REWARD " + str(self.curr_reward / 10))
                self.mean_rewards.append(self.curr_reward / 10)
                self.curr_reward = 0.0
            self.episode += 1
        return super()._on_step()


def save_video(model, env):
    images = []
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        img = env.render(mode="rgb_array")
        images.append(img)
        if done:
            break
    video_path = "rocketlander2.mp4"
    imageio.mimsave(video_path, [np.array(img) for i, img in enumerate(images)], fps=30)


def evaluate_model(model, env):
    num_landed = 0
    for _ in range(NUM_EVAL_EPISODES):
        obs = env.reset()
        cum_reward = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            cum_reward += rewards
            env.render()
            if done:
                print("REWARD: " + str(cum_reward))
                # if info["landed"]:
                #     print("landed")
                #     num_landed += 1
                break
    print(num_landed / NUM_EVAL_EPISODES)


def smooth_rewards(rewards, window=10):
    return np.convolve(rewards, np.ones(window) / window, mode="valid")

def main():
    env = gym.make("gym_rocketlander:rocketlander-v0")
    env.seed(42)
    env.reset()

    watch = not True

    if watch:
        model = PPO.load("base_model_2")
        evaluate_model(model, env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            ent_coef=ENTROPY,
        )
        callback = DataCallback()
        model.learn(total_timesteps=N_STEPS, callback=callback, progress_bar=False)
        evaluate_model(model, env)

        plt.plot(callback.mean_rewards)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Cumulative Reward")
        plt.show(block=True)
        model.save("base_model_2")

if __name__ == "__main__":
    main()
