import os
import gc
import torch
import pygame
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
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        
    def store(self, state, action, next_state, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)
        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return len(self.dones)
    
    
class DQN_Network(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQN_Network, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
        )
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        
    def forward(self, x):
        Q = self.FC(x)    
        return Q
    
        
class DQN_Agent:
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay, 
                  clip_grad_norm, learning_rate, discount, memory_capacity):
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.action_space = env.action_space
        self.action_space.seed(seed)
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)
        self.main_network = DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device)
        self.target_network = DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.clip_grad_norm = clip_grad_norm
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
                
    def select_action(self, state):
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()
        with torch.no_grad():
            Q_values = self.main_network(state)
            action = torch.argmax(Q_values).item()
            return action

    def learn(self, batch_size, done):
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        predicted_q = self.main_network(states)
        predicted_q = predicted_q.gather(dim=1, index=actions)
        with torch.no_grad():            
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0]
        next_target_q_value[dones] = 0
        y_js = rewards + (self.discount * next_target_q_value)
        loss = self.critertion(predicted_q, y_js)
        self.running_loss += loss.item()
        self.learned_counts += 1
        if done:
            episode_loss = self.running_loss / self.learned_counts
            self.loss_history.append(episode_loss)
            self.running_loss = 0
            self.learned_counts = 0
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step()

    def hard_update(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save(self, path):
        torch.save(self.main_network.state_dict(), path)
                  

class Model_TrainTest:
    def __init__(self, hyperparams):
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]
        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]
        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.memory_capacity = hyperparams["memory_capacity"]
        self.num_states = hyperparams["num_states"]
        self.map_size = hyperparams["map_size"]
        self.render_fps = hyperparams["render_fps"]
        self.env = gym.make('FrozenLake-v1', map_name=f"{self.map_size}x{self.map_size}", 
                            is_slippery=False, max_episode_steps=self.max_steps, 
                            render_mode="human" if self.render else None)
        self.env.metadata['render_fps'] = self.render_fps
        self.agent = DQN_Agent(env=self.env, 
                               epsilon_max=self.epsilon_max, 
                               epsilon_min=self.epsilon_min, 
                               epsilon_decay=self.epsilon_decay,
                               clip_grad_norm=self.clip_grad_norm,
                               learning_rate=self.learning_rate,
                               discount=self.discount_factor,
                               memory_capacity=self.memory_capacity)
                
    def state_preprocess(self, state:int, num_states:int):
        onehot_vector = torch.zeros(num_states, dtype=torch.float32, device=device)
        onehot_vector[state] = 1
        return onehot_vector
    
    def train(self): 
        total_steps = 0
        self.reward_history = []
        
        train_maps = [
            ["SFFF", "HFFF", "HFHH", "HFFG"],
            ["SFFF", "FHFH", "FHFF", "HFFG"],
            ["SHHH", "FHFF", "FFFF", "HFFG"],
            ["SFFH", "FHFF", "FFFF", "HFFG"],
            ["SFHH", "FHFF", "FFFF", "HHHG"],
        ]
        
        for episode in range(1, self.max_episodes+1):
            map_layout = train_maps[np.random.choice(len(train_maps))]
            self.env = gym.make('FrozenLake-v1', desc=map_layout, 
                                is_slippery=False, max_episode_steps=self.max_steps, 
                                render_mode="human" if self.render else None)
            self.env.metadata['render_fps'] = self.render_fps
            self.agent.action_space = self.env.action_space
            self.agent.observation_space = self.env.observation_space
            self.num_states = self.env.observation_space.n

            state, _ = self.env.reset(seed=seed)
            state = self.state_preprocess(state, num_states=self.num_states)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)
                next_state = self.state_preprocess(next_state, num_states=self.num_states)
                self.agent.replay_memory.store(state, action, next_state, reward, done)
                if len(self.agent.replay_memory) > self.batch_size and sum(self.reward_history) > 0:
                    self.agent.learn(self.batch_size, (done or truncation))
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()
                state = next_state
                episode_reward += reward
                step_size +=1
            self.reward_history.append(episode_reward)
            total_steps += step_size
            self.agent.update_epsilon()
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                print('\n~~~~~~Interval Save: Model saved.\n')
            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon_max:.2f}")
            print(result)
        self.plot_training(episode)
                                                                    
    def test(self, max_episodes):  
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()
        
        test_maps = [
            ["SFHH", "FFFH", "FHHH", "FFFG"],
            ["SFFH", "FFFF", "FHHF", "HHHG"]
        ]
        
        for episode in range(1, max_episodes+1):
            map_layout = test_maps[np.random.choice(len(test_maps))]
            self.env = gym.make('FrozenLake-v1', desc=map_layout, 
                                is_slippery=False, max_episode_steps=self.max_steps, 
                                render_mode="human" if self.render else None)
            self.env.metadata['render_fps'] = self.render_fps
            self.agent.action_space = self.env.action_space
            self.agent.observation_space = self.env.observation_space
            self.num_states = self.env.observation_space.n

            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
            while not done and not truncation:
                self.env.render()
                state = self.state_preprocess(state, num_states=self.num_states)
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                step_size += 1
            result = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Reward: {episode_reward:.2f}, ")
            print(result)
        pygame.quit()
    
    def plot_training(self, episode):
        sma = np.convolve(self.reward_history, np.ones(50)/50, mode='valid')
        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        if episode == self.max_episodes:
            plt.savefig('./reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close() 
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#CB291A', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        if episode == self.max_episodes:
            plt.savefig('./Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()        

if __name__ == '__main__':
    train_mode = False
    render = not train_mode
    map_size = 4
    RL_hyperparams = {
        "train_mode"            : train_mode,
        "RL_load_path"          : f'./{map_size}x{map_size}/final_weights' + '_' + '3000' + '.pth',
        "save_path"             : f'./{map_size}x{map_size}/final_weights',
        "save_interval"         : 1000,
        "clip_grad_norm"        : 3,
        "learning_rate"         : 6e-4,
        "discount_factor"       : 0.93,
        "batch_size"            : 32,
        "update_frequency"      : 10,
        "max_episodes"          : 3000 if train_mode else 5,
        "max_steps"             : 200,
        "render"                : render,
        "epsilon_max"           : 0.999 if train_mode else -1,
        "epsilon_min"           : 0.01,
        "epsilon_decay"         : 0.999,
        "memory_capacity"       : 4_000 if train_mode else 0,
        "map_size"              : map_size,
        "num_states"            : map_size ** 2,
        "render_fps"            : 6,
    }
    DRL = Model_TrainTest(RL_hyperparams)
    if train_mode:
        DRL.train()
    else:
        DRL.test(max_episodes = RL_hyperparams['max_episodes'])