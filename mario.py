import os
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the input to the first fully connected layer
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(256, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(240, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def store(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, input_channels, action_dim, epsilon_max, epsilon_min, epsilon_decay, 
                 learning_rate, discount, memory_capacity):
        self.input_channels = input_channels
        self.action_dim = action_dim
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.memory = ReplayMemory(memory_capacity)
        self.policy_net = QNetwork(input_channels, action_dim).to(device)
        self.target_net = QNetwork(input_channels, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon_max
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
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

def preprocess_state(state):
    state = state.transpose((2, 0, 1))  # Convert to CxHxW
    state = state / 255.0  # Normalize
    return state

def train_agent(env, agent, num_episodes, batch_size, target_update):
    rewards = []
    for episode in range(num_episodes):
        state = preprocess_state(env.reset()[0])
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            done = terminated or truncated
            agent.memory.store(state, action, next_state, reward, done)
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
        state = preprocess_state(env.reset()[0])
        total_reward = 0
        done = False
        while not done:
            env.render()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.policy_net(state).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = preprocess_state(next_state)
            done = terminated or truncated
            total_reward += reward
        total_rewards.append(total_reward)
        print(f'Episode {episode}, Total Reward: {total_reward}')
    env.close()
    return total_rewards

if __name__ == '__main__':
    env = gym.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    input_channels = 3  # Number of color channels in the input image
    action_dim = env.action_space.n
    
    agent = DQNAgent(input_channels, action_dim, epsilon_max=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                     learning_rate=0.001, discount=0.99, memory_capacity=10000)
    
    num_episodes = 500
    batch_size = 64
    target_update = 10
    
    rewards = train_agent(env, agent, num_episodes, batch_size, target_update)
    agent.save('mario_dqn.pth')
    
    agent.load('mario_dqn.pth')
    test_rewards = test_agent(env, agent, num_episodes=10)
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.show()