import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state)
        # print(f"State shape: {state.shape}, State dtype: {state.dtype}")
        state = state.to(device).unsqueeze(0)
        with torch.no_grad():
            action_mean, _ = self.policy_old(state)
        action = action_mean.cpu().data.numpy().flatten()
        return action

    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        for _ in range(self.K_epochs):
            action_means, state_values = self.policy(states)
            action_means_old, state_values_old = self.policy_old(states)
            state_values = state_values.squeeze()
            state_values_old = state_values_old.squeeze()

            advantages = rewards + self.gamma * state_values * (1 - dones) - state_values_old
            advantages = advantages.detach()

            # Ensure the dimensions match
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(-1)
            actions = actions.long().clamp(0, action_means.size(1) - 1)  # Clip actions to valid range
            action_means = action_means.gather(1, actions).squeeze(-1)
            action_means_old = action_means_old.gather(1, actions).squeeze(-1)

            ratios = torch.exp(action_means - action_means_old)
            advantages = advantages.unsqueeze(-1)  # Ensure advantages has the same dimension as ratios
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = []


def train_agent(env, agent, num_episodes, max_timesteps):
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update()
        rewards.append(total_reward)
        if episode % 10 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')
    return rewards

def test_agent(env, agent, num_episodes, max_timesteps):
    agent.policy.eval()
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            env.render()
            action = agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
        print(f'Episode {episode}, Total Reward: {total_reward}')
    env.close()
    return total_rewards

if __name__ == '__main__':
    env = gym.make('Humanoid-v4', render_mode='human')
    env.env.model.opt.timestep = 0.0005

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2)
    
    num_episodes = 1000
    max_timesteps = 2000
    
    rewards = train_agent(env, agent, num_episodes, max_timesteps)
    torch.save(agent.policy.state_dict(), 'humanoid_ppo.pth')
    
    agent.policy.load_state_dict(torch.load('humanoid_ppo.pth'))
    test_rewards = test_agent(env, agent, num_episodes=10, max_timesteps=max_timesteps)
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.show()