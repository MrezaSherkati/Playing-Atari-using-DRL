import sys
import gym
import torch
import pylab
#import random
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#from torch.autograd import Variable
from utils import find_max_lifes
from utils import check_live
from utils import get_frame
from utils import get_init_state
from agent import Agent
#from agent import device
#from model import *
from config import EPISODES
from config import evaluation_reward_length
from config import render_breakout
from config import train_frame
from config import Update_target_network_frequency
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2


env = gym.make('BreakoutDeterministic-v4')
env.render()

number_lives = find_max_lifes(env)
state_size = env.observation_space.shape
action_size = 3
rewards, episodes = [], []

agent = Agent(action_size)
evaluation_reward = deque(maxlen=evaluation_reward_length)
frame = 0
memory_size = 0

for e in range(EPISODES):
    done = False
    score = 0

    history = np.zeros([4, 5, 84, 84], dtype=np.uint8)
    step = 0
    d = False
    state = env.reset()
    life = number_lives

    get_init_state(history, state)

    while not done:
        step += 1
        frame += 1
        if render_breakout:
            env.render()

        
        action = agent.get_action(np.float32(history[:, :4, :, :]) / 255.)
        #print(action)
        
        next_state, reward, done, info = env.step(action + 1)

        frame_next_state = get_frame(next_state)
        history[:, 4, :, :] = frame_next_state
        terminal_state = check_live(life, info['ale.lives'])

        life = info['ale.lives']
        r = np.clip(reward, -1, 1)

         
        agent.memory.push(deepcopy(frame_next_state), action, r, terminal_state)
        
        if(frame >= train_frame):
            agent.train_policy_net(frame)
            
            if(frame % Update_target_network_frequency)== 0:
                agent.update_target_net()
        score += reward
        history[:, :4, :, :] = history[:, 1:, :, :]

        if frame % 50000 == 0:
            print('now time : ', datetime.now())
            rewards.append(np.mean(evaluation_reward))
            episodes.append(e)
            pylab.plot(episodes, rewards, 'b')
            pylab.savefig("./save_graph/breakout_dqn.png")

        if done:
            evaluation_reward.append(score)
            
            print("episode:", e, "  score:", score, "  memory length:",
                  len(agent.memory), "  epsilon:", agent.epsilon, "   steps:", step,
                  "    evaluation reward:", np.mean(evaluation_reward))

            
            if np.mean(evaluation_reward) > 10:
                torch.save(agent.model, "./save_model/breakout_dqn")
                sys.exit()