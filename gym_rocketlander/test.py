from envs.rocket_lander_env import RocketLander
import gym
import random

env = gym.make("gym_rocketlander:rocketlander-v0")
env.reset()
print(env.observation_space.sample())


class random_agent:
    def __init__(self, env):
        self.env = env
        self.actions = {}

    def random_action(self):
        return env.action_space.sample()
    


def main():
    # agent = random_agent(env)
    # total_reward = 0
    # while True:
    #     state, reward, done, _ = env.step(agent.random_action())
    #     total_reward += reward
    #     env.render()
    #     if done:
    #         break

    # print(f"total reward: {total_reward}")
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.show()


if __name__ == "__main__":
    main()
