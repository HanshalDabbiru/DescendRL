import gym
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


LEARNING_RATE = 0.0003
N_STEPS = 50000
BATCH_SIZE = 256
GAMMA = 1
NUM_EVAL_EPISODES = 20


class DataCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rewards = []
        self.curr_reward = 0

    def _on_step(self):
        step_reward = self.locals["rewards"][0].item()
        self.curr_reward += step_reward
        if self.locals["dones"][0]:
            self.rewards.append(self.curr_reward)
            print("CURR_REWARD " + str(self.curr_reward))
            self.curr_reward = 0
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
    video_path = "rocketlander.mp4"
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
                if info["landed"]:
                    print("landed")
                    num_landed += 1
                break
    print(num_landed / NUM_EVAL_EPISODES)


def smooth_rewards(rewards, window=10):
    return np.convolve(rewards, np.ones(window) / window, mode="valid")


def main():
    env = gym.make("gym_rocketlander:rocketlander-v0")
    env.seed(42)
    env.reset()

    # model = PPO("MlpPolicy", env, verbose=0, learning_rate=LEARNING_RATE, ent_coef=0.1)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=LEARNING_RATE)
    callback = DataCallback()
    model.learn(total_timesteps=N_STEPS, callback=callback, progress_bar=False)
    # model = PPO.load("contact_changed_model")

    evaluate_model(model, env)
    # save_video(model, env)

    model.save("contact_changed_model")

    plt.plot(callback.rewards)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Cumulative Reward")
    plt.show(block=True)


if __name__ == "__main__":
    main()
