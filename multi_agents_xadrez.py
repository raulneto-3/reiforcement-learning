import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from pettingzoo.classic import chess_v6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        self.memory = []

    def select_action(self, state, legal_moves):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        action_probs = action_probs.cpu().numpy().flatten()
        legal_moves_indices = [i for i, move in enumerate(legal_moves) if move]
        legal_action_probs = action_probs[legal_moves_indices]
        if legal_action_probs.sum() == 0:
            legal_action_probs = np.ones_like(legal_action_probs) / len(legal_action_probs)
        else:
            legal_action_probs /= legal_action_probs.sum()
        action = np.random.choice(legal_moves_indices, p=legal_action_probs)
        return action

    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) == 0:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        action_probs, state_values = self.policy(states)
        _, next_state_values = self.policy(next_states)
        state_values = state_values.squeeze()
        next_state_values = next_state_values.squeeze()

        advantages = rewards + self.gamma * next_state_values * (1 - dones) - state_values
        advantages = advantages.detach()

        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
        actor_loss = -(action_log_probs * advantages).mean()
        critic_loss = self.MseLoss(state_values, rewards + self.gamma * next_state_values * (1 - dones))

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []

def train_agent(env, agent1, agent2, num_episodes, max_timesteps):
    rewards = []
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        timestep = 0
        for agent in env.agent_iter():
            if timestep >= max_timesteps:
                break
            observation, reward, done, truncation, _ = env.last(observe=True)
            if done or truncation:
                action = None
            else:
                state = observation['observation'].flatten()
                legal_moves = observation['action_mask']
                action = agent1.select_action(state, legal_moves) if agent == 'player_0' else agent2.select_action(state, legal_moves)
            env.step(action)
            next_observation, next_reward, next_done, next_truncation, _ = env.last(observe=True)
            next_state = next_observation['observation'].flatten()
            agent1.store_transition((state, action, reward, next_state, done)) if agent == 'player_0' else agent2.store_transition((state, action, reward, next_state, done))
            total_reward += reward
            timestep += 1
            if next_done or next_truncation:
                break
        agent1.update()
        agent2.update()
        rewards.append(total_reward)
        if episode % 10 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')
    return rewards

def test_agent(env, agent1, agent2, num_episodes, max_timesteps):
    agent1.policy.eval()
    agent2.policy.eval()
    total_rewards = []
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        timestep = 0
        for agent in env.agent_iter():
            if timestep >= max_timesteps:
                break
            observation, reward, done, truncation, _ = env.last(observe=True)
            if done or truncation:
                action = None
            else:
                state = observation['observation'].flatten()
                legal_moves = observation['action_mask']
                action = agent1.select_action(state, legal_moves) if agent == 'player_0' else agent2.select_action(state, legal_moves)
            env.step(action)
            next_observation, next_reward, next_done, next_truncation, _ = env.last(observe=True)
            total_reward += reward
            timestep += 1
            if next_done or next_truncation:
                break
        total_rewards.append(total_reward)
        print(f'Episode {episode}, Total Reward: {total_reward}')
    return total_rewards

def display_game(env, agent1, agent2, max_timesteps):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, truncation, _ = env.last(observe=True)
        if done or truncation:
            action = None
        else:
            state = observation['observation'].flatten()
            legal_moves = observation['action_mask']
            action = agent1.select_action(state, legal_moves) if agent == 'player_0' else agent2.select_action(state, legal_moves)
        env.step(action)
        print(env.render(mode='ansi'))
        if env.dones[agent] or env.truncations[agent]:
            break

if __name__ == '__main__':
    env = chess_v6.env(render_mode='human')
    env.reset()

    state_dim = env.observation_space(env.agents[0])['observation'].shape[0] * env.observation_space(env.agents[0])['observation'].shape[1] * env.observation_space(env.agents[0])['observation'].shape[2]
    action_dim = env.action_space(env.agents[0]).n
    
    agent1 = A2CAgent(state_dim, action_dim, lr=0.0003, gamma=0.99)
    agent2 = A2CAgent(state_dim, action_dim, lr=0.0003, gamma=0.99)
    
    num_episodes = 1000
    max_timesteps = 100
    
    rewards = train_agent(env, agent1, agent2, num_episodes, max_timesteps)
    torch.save(agent1.policy.state_dict(), 'chess_agent1_a2c.pth')
    torch.save(agent2.policy.state_dict(), 'chess_agent2_a2c.pth')
    
    agent1.policy.load_state_dict(torch.load('chess_agent1_a2c.pth'))
    agent2.policy.load_state_dict(torch.load('chess_agent2_a2c.pth'))
    test_rewards = test_agent(env, agent1, agent2, num_episodes=1, max_timesteps=500)
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.show()
    
    print("Displaying a complete game between the agents:")
    display_game(env, agent1, agent2, max_timesteps=100)