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
from gymnasium import spaces
import pygame

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

class MazeEnv(gym.Env):
    def __init__(self, maze):
        super(MazeEnv, self).__init__()
        self.maze = maze
        self.start_pos = (0, 0)
        self.goal_pos = (len(maze) - 1, len(maze[0]) - 1)
        self.current_pos = self.start_pos
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(maze), len(maze[0])), dtype=np.float32)
        self.reset()
        self.window_size = 600
        self.cell_size = self.window_size // len(maze)
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Maze")

    def reset(self):
        self.current_pos = self.start_pos
        return self._get_obs()

    def step(self, action):
        next_pos = self._move(action)
        if self._is_valid(next_pos):
            self.current_pos = next_pos

        reward = 1 if self.current_pos == self.goal_pos else -0.1
        done = self.current_pos == self.goal_pos
        return self._get_obs(), reward, done, False, {}

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))
        for row in range(len(self.maze)):
            for col in range(len(self.maze[0])):
                color = (0, 0, 255) if self.maze[row, col] == 1 else (255, 255, 255)
                pygame.draw.rect(self.screen, color, pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(self.goal_pos[1] * self.cell_size, self.goal_pos[0] * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.circle(self.screen, (255, 0, 0), (self.current_pos[1] * self.cell_size + self.cell_size // 2, self.current_pos[0] * self.cell_size + self.cell_size // 2), self.cell_size // 3)
        pygame.display.flip()

    def _get_obs(self):
        obs = np.zeros_like(self.maze, dtype=np.float32)
        obs[self.current_pos] = 1
        return obs

    def _move(self, action):
        if action == 0:  # up
            return (self.current_pos[0] - 1, self.current_pos[1])
        elif action == 1:  # down
            return (self.current_pos[0] + 1, self.current_pos[1])
        elif action == 2:  # left
            return (self.current_pos[0], self.current_pos[1] - 1)
        elif action == 3:  # right
            return (self.current_pos[0], self.current_pos[1] + 1)

    def _is_valid(self, pos):
        return (0 <= pos[0] < len(self.maze)) and (0 <= pos[1] < len(self.maze[0])) and (self.maze[pos] == 0)

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
        x = x.view(x.size(0), -1)  # Achatar a entrada
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
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
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
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.policy_net(state).argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
        total_rewards.append(total_reward)
        print(f'Episode {episode}, Total Reward: {total_reward}')
    env.close()
    return total_rewards

if __name__ == '__main__':
    # Define the maze (0 = free space, 1 = wall)
    maze = np.array([
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ])
    
    env = MazeEnv(maze)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    
    agent = DQN_Agent(state_size, action_size, epsilon_max=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                      learning_rate=0.001, discount=0.99, memory_capacity=10000)
    
    num_episodes = 500
    batch_size = 64
    target_update = 10
    
    # rewards = train_agent(env, agent, num_episodes, batch_size, target_update)
    # agent.save('maze_dqn.pth')
    
    agent.load('maze_dqn.pth')
    test_rewards = test_agent(env, agent, num_episodes=10)
    
    # plt.plot(rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('Training Rewards')
    # plt.show()