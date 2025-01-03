import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

class QLearningAgent:
    def __init__(self, state_size, action_size, epsilon_max, epsilon_min, epsilon_decay, 
                 learning_rate, discount, memory_capacity):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.memory = ReplayMemory(memory_capacity)
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
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

def train_agent(agent, mazes, num_episodes, batch_size, target_update):
    rewards = []
    for episode in range(num_episodes):
        maze = random.choice(mazes)
        state = (0, 0)
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = step(maze, state, action)
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

def step(maze, state, action):
    x, y = state
    if action == 0:  # up
        x = max(x - 1, 0)
    elif action == 1:  # right
        y = min(y + 1, maze.shape[1] - 1)
    elif action == 2:  # down
        x = min(x + 1, maze.shape[0] - 1)
    elif action == 3:  # left
        y = max(y - 1, 0)
    
    next_state = (x, y)
    if maze[next_state] == 1:
        return state, -1, False  # hit a wall
    if next_state == (maze.shape[0] - 1, maze.shape[1] - 1):
        return next_state, 10, True  # reached the goal
    return next_state, -0.1, False  # valid move

def load_mazes(directory):
    mazes = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            maze = np.load(os.path.join(directory, filename))
            mazes.append(maze)
    return mazes

def render_maze(screen, maze, state, cell_size=20):
    colors = {
        0: (255, 255, 255),  # path
        1: (0, 0, 0),        # wall
        'agent': (0, 0, 255),# agent
        'goal': (0, 255, 0)  # goal
    }
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            color = colors[maze[x, y]]
            pygame.draw.rect(screen, color, pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size))
    agent_x, agent_y = state
    pygame.draw.rect(screen, colors['agent'], pygame.Rect(agent_y * cell_size, agent_x * cell_size, cell_size, cell_size))
    goal_x, goal_y = maze.shape[0] - 1, maze.shape[1] - 1
    pygame.draw.rect(screen, colors['goal'], pygame.Rect(goal_y * cell_size, goal_x * cell_size, cell_size, cell_size))
    pygame.display.flip()

def test_agent(agent, mazes, num_episodes):
    pygame.init()
    cell_size = 20
    screen = pygame.display.set_mode((mazes[0].shape[1] * cell_size, mazes[0].shape[0] * cell_size))
    pygame.display.set_caption('Maze Navigation')
    
    for episode in range(num_episodes):
        maze = random.choice(mazes)
        state = (0, 0)
        total_reward = 0
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            render_maze(screen, maze, state, cell_size)
            action = agent.select_action(state)
            next_state, reward, done = step(maze, state, action)
            state = next_state
            total_reward += reward
            pygame.time.wait(100)
        print(f'Episode {episode}, Total Reward: {total_reward}')
    pygame.quit()

if __name__ == '__main__':
    directory = 'mazes'
    mazes = load_mazes(directory)
    state_size = 2  # (x, y) position
    action_size = 4  # up, right, down, left
    
    agent = QLearningAgent(state_size, action_size, epsilon_max=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                           learning_rate=0.001, discount=0.99, memory_capacity=10000)
    
    num_episodes = 500
    batch_size = 64
    target_update = 10
    
    rewards = train_agent(agent, mazes, num_episodes, batch_size, target_update)
    agent.save('maze_agent.pth')
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.show()
    
    agent.load('maze_agent.pth')
    test_agent(agent, mazes, num_episodes=10)