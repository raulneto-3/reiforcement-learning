import os
import gc
import random
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Used for debugging; CUDA related errors shown immediately.

# Seed everything for reproducible results
seed = 2024
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def store(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return len(self.memory)

class DQN_Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN_Network, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN_Agent:
    def __init__(self, state_size, action_size, epsilon_max, epsilon_min, epsilon_decay, 
                 learning_rate, discount, memory_capacity):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.memory = ReplayMemory(memory_capacity)
        self.policy_net = DQN_Network(state_size, action_size).to(device)
        self.target_net = DQN_Network(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon_max
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
    
    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        states, actions, next_states, rewards, dones = self.memory.sample(batch_size)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (self.discount * next_q_values * (1 - dones))
        
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_agent(env, agent, num_episodes, batch_size, target_update):
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncation, _ = env.step(action)
            agent.memory.store(state, action, next_state, reward, done or truncation)
            agent.learn(batch_size)
            state = next_state
            total_reward += reward
        agent.update_epsilon()
        rewards.append(total_reward)
        if episode % target_update == 0:
            agent.update_target_network()
        if episode % 10 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}')
    return rewards

def test_agent(env, agent, num_episodes):
    agent.policy_net.eval()
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.policy_net(state).argmax().item()
            next_state, reward, done, truncation, _ = env.step(action)
            state = next_state
            total_reward += reward
        total_rewards.append(total_reward)
        print(f'Episode {episode}, Total Reward: {total_reward}')
    env.close()
    return total_rewards

if __name__ == '__main__':
    # Create environment without rendering for training
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQN_Agent(state_size, action_size, epsilon_max=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                      learning_rate=0.001, discount=0.99, memory_capacity=10000)
    
    num_episodes = 500
    batch_size = 64
    target_update = 10
    
    rewards = train_agent(env, agent, num_episodes, batch_size, target_update)
    agent.save('cartpole_dqn.pth')
    
    # Create environment with rendering for testing
    env = gym.make('CartPole-v1', render_mode='human')
    agent.load('cartpole_dqn.pth')
    test_rewards = test_agent(env, agent, num_episodes=10)
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.show()