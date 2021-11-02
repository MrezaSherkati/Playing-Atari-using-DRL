import random
import numpy as np

from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import *
from config import learning_rate
from config import batch_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, action_size):
        self.load_model = False

        self.action_size = action_size

        
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 20000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 10000
        self.update_target = 2

        
        self.memory = ReplayMemory()

        
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        self.target_net = DQN(action_size)
        self.target_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

        
        self.update_target_net()

        if self.load_model:
           self.policy_net = torch.load('save_model/breakout_dqn')

    
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
            
        else:
             
             with torch.no_grad():
                 
                 x = self.policy_net(state)
             
                 y = x[0]
                 
                 values, index = torch.max(y, 0)
                 
                 index = torch.tensor([index], device=device, dtype=torch.long)
                 
                 return index
             
            
            

    
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] 
        
        
        
        
        actions=torch.tensor(actions)
        actions=actions.unsqueeze(1)
        
        state_action_values = self.policy_net(states).gather(1,actions)
        
        #print(state_action_values)
        
        next_state_values = self.policy_net(next_states)
        
        
        for i in range(32):
            if dones[i] == 'true':
                next_state_values[i]=0
            
        
        
         
        #max_next_state = torch.max(self.target_net(next_states)).item()
        q_next_state=self.target_net(next_states)
        rewards=np.array(rewards)
        rewards = torch.from_numpy(rewards).double()
        expected_state_action_values=[]
        for i in range(32):
            if dones[i] == 'true':
                expected_state_action_values.append(rewards[i])
                
            else:
                temp=rewards[i]+ self.discount_factor*torch.max(q_next_state[i])
                expected_state_action_values.append(temp)
                
        
        
        #expected_state_action_values = (max_next_state * self.discount_factor) + rewards
        
        #print(expected_state_action_values)
        expected_state_action_values=np.asarray(expected_state_action_values)
        expected_state_action_values=expected_state_action_values.reshape((32,1))
        expected_state_action_values=expected_state_action_values.tolist()
        expected_state_action_values=torch.tensor(expected_state_action_values)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
         
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
          param.grad.data.clamp_(-1, 1)
        self.optimizer.step()