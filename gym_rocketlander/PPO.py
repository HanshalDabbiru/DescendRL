import gym
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

LEARNING_RATE = 0.03
N_STEPS = 10000
BATCH_SIZE = 64
GAMMA = 0.99
NUM_EVAL_EPISODES = 100

class DataCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rewards = []
        self.curr_reward = 0

    def _on_step(self):
        step_reward = self.locals["rewards"]
        self.curr_reward += step_reward
        if self.locals["dones"]:
            self.rewards.append(self.curr_reward)
            self.curr_reward = 0
        return super()._on_step()

def save_video(model):
    images = []
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        img = env.render(mode='rgb_array')
        images.append(img) 
        if done:
            break
    video_path = "rocketlander.mp4"
    imageio.mimsave(video_path, [np.array(img) for i, img in enumerate(images)], fps=30)

def evaluate_model(model):
    num_landed = 0
    for _ in range(NUM_EVAL_EPISODES):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            if done:
                print(info["landed"])
                if info["landed"]:
                    num_landed += 1
                break
    print(num_landed / NUM_EVAL_EPISODES)

env = gym.make("gym_rocketlander:rocketlander-v0")
env.reset()

model = PPO("MlpPolicy", env, verbose=0)
callback = DataCallback()
model.learn(total_timesteps=N_STEPS, callback=callback)

evaluate_model(model)
save_video(model)

plt.plot(callback.rewards)
plt.xlabel("Number of Episodes")
plt.ylabel("Cumulative Reward")
plt.show(block=True)