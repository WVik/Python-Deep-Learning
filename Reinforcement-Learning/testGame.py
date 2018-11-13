import gym
import random
import numpy as np
import tflearn
import tflearn.layers.code import input_data, dropout, fully_connected

from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-4
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 300
score_requirement = 50
initial_games = 10000

def first_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation,reward,done,info = env.step(action)
            if done:
                break

first_games()