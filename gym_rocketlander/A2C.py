import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
LEARNING_RATE = 3e-4
ENTROPY_BETA = 0.01
NUM_EPOCHS = 10
BATCH_SIZE = 64
UPDATE_STEPS = 2048


# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.policy_layer = nn.Linear(128, action_dim)  # Actor
        self.value_layer = nn.Linear(128, 1)  # Critic

    def forward(self, state):
        x = self.shared_layer(state)
        policy_logits = self.policy_layer(x)
        value = self.value_layer(x)
        return policy_logits, value


# Helper Functions
def compute_gae(rewards, dones, values, next_value, gamma=GAMMA, lam=LAMBDA):
    """Compute Generalized Advantage Estimation (GAE)."""
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (1 - dones[i]) * next_value - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
    return advantages


def compute_returns(rewards, dones, next_value, gamma=GAMMA):
    """Compute discounted returns."""
    returns = []
    R = next_value
    for r, done in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - done)
        returns.insert(0, R)
    return returns


# Initialize environment and model
env = gym.make("gym_rocketlander:rocketlander-v0")
state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

model = ActorCritic(state_dim, 3)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for episode in range(1000):  # Number of training episodes
    states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []
    state = env.reset()
    done = False
    total_reward = 0

    # Collect trajectories
    for step in range(UPDATE_STEPS):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean, value = model(state_tensor)
        std = torch.exp(value)


        # Sample action
        action_probs = torch.softmax(mean, dim=-1)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = action.detach().numpy()
        # print(action.item())
        next_state, reward, done, _ = env.step(action[0])
        env.render()
        total_reward += reward

        # Store trajectory data
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        old_log_probs.append(dist.log_prob(torch.FloatTensor(action)).sum().item())
        values.append(value.item())

        state = next_state

        if done:
            state = env.reset()

    # Compute GAE and returns
    next_value = model(torch.FloatTensor(state).unsqueeze(0))[1].item()
    advantages = compute_gae(rewards, dones, values, next_value)
    returns = compute_returns(rewards, dones, next_value)
    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)

    # Update policy and value network
    for epoch in range(NUM_EPOCHS):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_indices = indices[start:end]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_old_log_probs = batch_old_log_probs.squeeze(-1)

            # Forward pass
            mean, log_std, values = model(batch_states)
            values = values.squeeze()
            action_probs = torch.softmax(mean, dim=-1)
            dist = torch.distributions.Normal(mean, std)

            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            entropy = dist.entropy().mean()

            # PPO Clipped Objective
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            policy_loss = -torch.mean(
                torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages)
            )

            # Value loss
            value_loss = torch.mean((values - batch_returns) ** 2)

            # Combined loss
            loss = policy_loss + 0.5 * value_loss - ENTROPY_BETA * entropy

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Logging
    total_reward = sum(rewards)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
