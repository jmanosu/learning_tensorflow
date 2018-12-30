import sys
import gym, numpy, random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from collections import Counter



LR = 1e-2
env = gym.make('LunarLanderContinuous-v2');
initial_games = 5000
gamma = .95
alpha = 0.1
epsilon = 0.1
total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5



class model():
    def __init__(self, i_size, h_size, a_size):
        self.input_layer = tf.placeholder(shape=[None, i_size], dtype=tf.float32)
        self.hidden_layer = slim.fully_connected(self.input_layer, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output_layer = slim.fully_connected(self.hidden_layer, a_size, activation_fn=tf.nn.softmax,biases_initializer=None)
        self.action = tf.argmax(self.output_layer,1)

        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output_layer)[0]) * tf.shape(self.output_layer)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output_layer, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        #self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        self.update_batch = tf.train.AdamOptimizer(learning_rate=LR).minimize(self.loss)



'''
def get_action(action):
    return {
        0: [0,0],
        1: [1,0],
        2: [0,.5],
        3: [0,-.5],
        4: [1,.5],
        5: [1,-.5],
        6: [1,1],
        7: [1,-1],
    }[action]

def get_tensor_output(action):
    return {
        0: [1,0,0,0,0,0,0,0],
        1: [0,1,0,0,0,0,0,0],
        2: [0,0,1,0,0,0,0,0],
        3: [0,0,0,1,0,0,0,0],
        4: [0,0,0,0,1,0,0,0],
        5: [0,0,0,0,0,1,0,0],
        6: [0,0,0,0,0,0,1,0],
        7: [0,0,0,0,0,0,0,1],
    }[action]
'''
def get_action(action):
    return {
        0: [0,0],
        1: [1,0],
        2: [0,1],
        3: [0,-1],
    }[action]


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def main():
    tf.reset_default_graph()
    myModel = model(i_size=8,h_size=32,a_size=4)
    init = tf.global_variables_initializer()
    render = False
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        total_reward = []
        total_lenght = []
        ep_history = []
        while i < total_episodes:
            s = env.reset()
            running_reward = 0
            for j in range(max_ep):
                #Probabilistically pick an action given our network outputs.
                a_dist = sess.run(myModel.output_layer,feed_dict={myModel.input_layer:[s]})
                a = np.random.choice(a_dist[0],p=a_dist[0])
                action = np.argmax(a_dist == a)
                a = get_action(action)
                #if np.random.rand(1) < epsilon:
                #    a = get_action(a_dist[0])
                #else:
                #    a = get_action(random.randint(0,3))
                #a = np.random.choice(a_dist[0],p=a_dist[0])
                #a = np.argmax(a_dist == a)
                #ta = get_tensor_output(a_dist)
                s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
                ep_history.append([s,action,r,s1])
                s = s1
                running_reward += r
                if render:
                    env.render()
                if r == 100:
                    print('completed')
                    render = True
                if d == True and i % update_frequency == 0 and i != 0:
                    #Update the network.
                    ep_history = np.array(ep_history)
                    ep_history[:,2] = discount_rewards(ep_history[:,2])
                    feed_dict={myModel.reward_holder:ep_history[:,2],
                            myModel.action_holder:ep_history[:,1],myModel.input_layer:np.vstack(ep_history[:,0])}
                    _ = sess.run(myModel.update_batch, feed_dict=feed_dict)

                    total_reward.append(running_reward)
                    total_lenght.append(j)

                    if i % 100 == 0:
                        print(np.mean(total_reward[-100:]))

                    ep_history = []

                    break
                elif d == True:
                    break

            i += 1


main()
