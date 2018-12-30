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
initial_games = 5000

def random_games(minimum_score):
    training_data = []
    mean = 0
    for i_episode in range(initial_games):
        observation = env.reset()
        game_data = []
        score = 0
        prev_observation = []
        for _ in range(100):
            choice = random.randint(0,3)
            action_data = []
            action = []
            if choice == 0:
                action = [0,0]
                action_data = [1,0,0,0]
            elif choice == 1:
                action = [0.5,0]
                action_data = [0,1,0,0]
            elif choice == 2:
                action = [0,0.5]
                action_data = [0,0,1,0]
            elif choice == 3:
                action = [0,-0.5]
                action_data = [0,0,0,1]
            observation, reward, done, info = env.step(action)
            score += reward

            if len(prev_observation) > 0:
                game_data.append([prev_observation, action_data])

            prev_observation = observation
            if done:
                break
        mean += score
        if score > minimum_score:
            print(score)
            for new_data in game_data:
                training_data.append(new_data)
    print("Average score: ", mean/initial_games)
    return np.array(training_data)

def create_model(input_size, output_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    Y = [i[1] for i in training_data]

    if not model:
        model = create_model(input_size = len(X[0]), output_size = len(Y[0]))

    print("X length ", len(X), " X width ", len(X[0]))
    print("Y length ", len(Y), " Y width ", len(Y[0]))
    model.fit({'input': X}, {'targets': Y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')


    return model

def main():
    data = random_games(10)
    model = train_model(data)
    observation = []
    for i in range(10):
        env.reset()
        for i in range(50):
            env.render()
            if len(observation) == 0:
                action = [round(random.uniform(-1,1), 1),round(random.uniform(-1,1), 1)]
                observation, reward, done, info = env.step(action)
            else:
                prediction = np.argmax(model.predict(observation.reshape(1,8,1)))
                action = []
                if prediction == 0:
                    action = [0,0]
                elif prediction == 1:
                    action = [0.5,0]
                elif prediction == 2:
                    action = [0,0.5]
                elif prediction == 3:
                    action = [0,-0.5]
                observation, reward, done, info = env.step(action)
            if done:
                break

main()
