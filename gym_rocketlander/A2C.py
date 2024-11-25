import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# # Hyperparameters
# GAMMA = 0.99
# LAMBDA = 0.95
# CLIP_EPSILON = 0.2
# LEARNING_RATE = 3e-4
# ENTROPY_BETA = 0.01
# NUM_EPOCHS = 10
# BATCH_SIZE = 64
# UPDATE_STEPS = 2048


# # Define the Actor-Critic Network
# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#         self.shared_layer = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
#         self.policy_layer = nn.Linear(128, action_dim)  # Actor
#         self.value_layer = nn.Linear(128, 1)  # Critic

#     def forward(self, state):
#         x = self.shared_layer(state)
#         policy_logits = self.policy_layer(x)
#         value = self.value_layer(x)
#         return policy_logits, value


# # Helper Functions
# def compute_gae(rewards, dones, values, next_value, gamma=GAMMA, lam=LAMBDA):
#     """Compute Generalized Advantage Estimation (GAE)."""
#     advantages = []
#     gae = 0
#     for i in reversed(range(len(rewards))):
#         delta = rewards[i] + gamma * (1 - dones[i]) * next_value - values[i]
#         gae = delta + gamma * lam * (1 - dones[i]) * gae
#         advantages.insert(0, gae)
#         next_value = values[i]
#     return advantages


# def compute_returns(rewards, dones, next_value, gamma=GAMMA):
#     """Compute discounted returns."""
#     returns = []
#     R = next_value
#     for r, done in zip(reversed(rewards), reversed(dones)):
#         R = r + gamma * R * (1 - done)
#         returns.insert(0, R)
#     return returns


# # Initialize environment and model
# env = gym.make("gym_rocketlander:rocketlander-v0")
# state_dim = env.observation_space.shape[0]
# # action_dim = env.action_space.n

# model = ActorCritic(state_dim, 3)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # Training Loop
# for episode in range(1000):  # Number of training episodes
#     states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []
#     state = env.reset()
#     done = False
#     total_reward = 0

#     # Collect trajectories
#     for step in range(UPDATE_STEPS):
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
#         mean, value = model(state_tensor)
#         std = torch.exp(value)


#         # Sample action
#         action_probs = torch.softmax(mean, dim=-1)
#         dist = torch.distributions.Normal(mean, std)
#         action = dist.sample()
#         action = action.detach().numpy()
#         # print(action.item())
#         next_state, reward, done, _ = env.step(action[0])
#         env.render()
#         total_reward += reward

#         # Store trajectory data
#         states.append(state)
#         actions.append(action)
#         rewards.append(reward)
#         dones.append(done)
#         old_log_probs.append(dist.log_prob(torch.FloatTensor(action)).sum().item())
#         values.append(value.item())

#         state = next_state

#         if done:
#             state = env.reset()

#     # Compute GAE and returns
#     next_value = model(torch.FloatTensor(state).unsqueeze(0))[1].item()
#     advantages = compute_gae(rewards, dones, values, next_value)
#     returns = compute_returns(rewards, dones, next_value)
#     advantages = torch.FloatTensor(advantages)
#     returns = torch.FloatTensor(returns)
#     states = torch.FloatTensor(states)
#     actions = torch.FloatTensor(actions)
#     old_log_probs = torch.FloatTensor(old_log_probs)

#     # Update policy and value network
#     for epoch in range(NUM_EPOCHS):
#         indices = np.random.permutation(len(states))
#         for start in range(0, len(states), BATCH_SIZE):
#             end = start + BATCH_SIZE
#             batch_indices = indices[start:end]

#             batch_states = states[batch_indices]
#             batch_actions = actions[batch_indices]
#             batch_advantages = advantages[batch_indices]
#             batch_returns = returns[batch_indices]
#             batch_old_log_probs = old_log_probs[batch_indices]
#             batch_old_log_probs = batch_old_log_probs.squeeze(-1)

#             # Forward pass
#             mean, log_std, values = model(batch_states)
#             values = values.squeeze()
#             action_probs = torch.softmax(mean, dim=-1)
#             dist = torch.distributions.Normal(mean, std)

#             new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
#             entropy = dist.entropy().mean()

#             # PPO Clipped Objective
#             ratio = torch.exp(new_log_probs - batch_old_log_probs)
#             clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
#             policy_loss = -torch.mean(
#                 torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages)
#             )

#             # Value loss
#             value_loss = torch.mean((values - batch_returns) ** 2)

#             # Combined loss
#             loss = policy_loss + 0.5 * value_loss - ENTROPY_BETA * entropy

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     # Logging
#     total_reward = sum(rewards)
#     print(f"Episode {episode + 1}: Total Reward = {total_reward}")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Hyperparameters
GAMMA = 0.99  # Discount factor
LAMBDA = 0.95  # GAE parameter
CLIP_EPSILON = 0.2  # PPO clipping parameter
LEARNING_RATE = 3e-4  # Learning rate for optimizer
NUM_EPOCHS = 10  # Number of training epochs
BATCH_SIZE = 64  # Mini-batch size
UPDATE_STEPS = 2048  # Steps to collect trajectories before an update
VALUE_LOSS_COEFF = 0.5  # Weight for value loss
ENTROPY_COEFF = 0.01  # Weight for entropy bonus


# Define Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared network for feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        # Actor network outputs mean of Gaussian
        self.actor_mean = nn.Linear(64, action_dim)
        # Learnable standard deviation (log scale)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        # Critic network outputs value of state
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor_mean(features)
        value = self.critic(features)
        return mean, self.actor_log_std, value


# Function to calculate GAE (Generalized Advantage Estimation)
def compute_gae(rewards, dones, values, gamma, lam):
    advantages = []
    advantage = 0
    next_value = 0  # Bootstrap value after the final step

    for t in reversed(range(len(rewards))):
        # Temporal difference error
        td_error = rewards[t] + (1 - dones[t]) * gamma * next_value - values[t]
        # GAE advantage calculation
        advantage = td_error + (1 - dones[t]) * gamma * lam * advantage
        next_value = values[t]
        advantages.insert(0, advantage)  # Insert at the start of the list

    # Return advantages and computed returns (advantage + value)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)


# PPO Training Loop
def train_ppo(env, model, optimizer):
    for episode in range(1000):  # Train for 1000 episodes
        states, actions, rewards, dones, values, old_log_probs = [], [], [], [], [], []
        state = env.reset()
        total_reward = 0

        # Step 1: Collect trajectories
        for _ in range(UPDATE_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, log_std, value = model(state_tensor)
            std = torch.exp(log_std)  # Convert log_std to std

            # Sample action from a Gaussian distribution
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()  # Sample action
            action = action[0]
            log_prob = dist.log_prob(action).sum(dim=-1)  # Log probability of action

            # Interact with the environment
            next_state, reward, done, _ = env.step(action.detach().numpy())
            env.render()
            total_reward += reward

            # Store trajectory data
            states.append(state)
            actions.append(action.detach().numpy())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            old_log_probs.append(log_prob.item())

            state = next_state
            if done:
                state = env.reset()

        # Step 2: Compute advantages and returns
        advantages, returns = compute_gae(rewards, dones, values, GAMMA, LAMBDA)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Step 3: Update policy and value networks
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)

        for _ in range(NUM_EPOCHS):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]

                # Mini-batches
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                mean, log_std, values = model(batch_states)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)

                # Compute log probabilities for current policy
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()  # Encourage exploration

                # PPO objective: clipped surrogate loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss: MSE between returns and predicted values
                value_loss = ((batch_returns - values.squeeze()) ** 2).mean()

                # Combine losses
                loss = (
                    actor_loss + VALUE_LOSS_COEFF * value_loss - ENTROPY_COEFF * entropy
                )

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


# Initialize environment and model
env = gym.make("gym_rocketlander:rocketlander-v0")
env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Instantiate model and optimizer
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train PPO agent
train_ppo(env, model, optimizer)
