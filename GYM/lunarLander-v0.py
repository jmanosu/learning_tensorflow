import sys
sys.path.append("/home/jmanosu/gym")
import gym, numpy, random
import tensorflow as tf
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from collections import Counter

LR = 1e-3
env = gym.make('LunarLanderContinuous-v2');
observation = env.reset()
initial_games = 20

def random_games(minimum_score):
    training_data = []
    for i_episode in range(initial_games):
        observation = env.reset()
        game_data = []
        score = 0
        prev_observation = []
        for _ in range(60):
            action = [round(random.uniform(-1,1), 1),round(random.uniform(-1,1), 1)]
            observation, reward, done, info = env.step(action)
            score += reward

            if len(prev_observation) > 0:
                game_data.append([prev_observation, action])

            prev_observation = observation
            if done:
                break
        if score > minimum_score:
            for new_data in game_data:
                training_data.append(new_data)

    return np.array(training_data)

temp = random_games(-20)
print(temp)
