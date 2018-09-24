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
env = gym.make('MountainCarContinuous-v0')
observation = env.reset()
goal_steps = 60
score_requirement = -20
initial_games = 20000
totalreward = 0

def random_games():
    maxobservation = -1.2
    for i_episode in range(1):
        observation = env.reset()
        score = 0
        game_memory = []
        totalreward = 0
        prev_obs = []
        for _ in range(goal_steps):
            env.render()
            action = [random.randint(-1,1)]
            observation, reward, done, info = env.step([1.0])
            if observation[0] > maxobservation:
                maxobservation = observation[0]
            if done:
                #break
                #print("Episode finished after {} timesteps".format(t+1))
                break
    print("maxobservation",maxobservation)

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        observation = env.reset()
        score = 0
        game_memory = []
        prev_observation = []
        maxposition = -1.2
        minposition = 0.6
        for _ in range(goal_steps):
            action = [random.randint(-1,1)]
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if maxposition < observation[0]:
                maxposition = observation[0]
            if minposition > observation[0]:
                minposition = observation[0]
            if done:
                break
            #working model filter
        if maxposition > -.4 and minposition < -.7:
            #original model training model
        #if maxposition > -.2 and minposition < -.7:
            accepted_scores.append([score])
            for data in game_memory:
                output = [0,0,0]
                if data[1] == [-1]:
                    output = [1,0,0]
                elif data[1] == [0]:
                    output = [0,1,0]
                elif data[1] == [1]:
                    output = [0,0,1]
                training_data.append([data[0], output])
    training_data_save = np.array(training_data)
    np.save('save.npy', training_data_save)
    #print('Average accepted scores:', np.mean(accepted_scores))
    #print('Median accepted score:', np.median(accepted_scores))
    #print(len(accepted_scores))

    return training_data


def neural_network_model(input_size):
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

    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(trianing_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    print(X)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')


    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []
min_position = 0.6
max_position = -1.2
pause = False


for i_episode in range(10):
    observation = env.reset()
    score = 0
    game_memory = []
    prev_obs = []
    train = False
    for _ in range(300):
        #if i_episode > 190:
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            if action == 0:
                action = -1
            elif action == 1:
                action = 0
            elif action == 2:
                action = 1
        choices.append(action)
        new_observation, reward, done, info = env.step([action])
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if new_observation[0] > max_position:
            max_position = new_observation[0]
            train = True
        elif new_observation[0] < min_position:
            min_position = new_observation[0]
            train = True
        if done:
            print(done)
            break
    scores.append(score)
