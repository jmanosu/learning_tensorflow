import gym, numpy, random
import tensorflow as tf
import numpy as np
from tflearn.layers.core import input_data
from tflearn.layers.estimator import regression
from statistics import mean, meadian
from collections import Counter


env = gym.make('CartPole-v0')
observation = env.reset()
totalreward = 0

W1 = tf.Variable(tf.random_normal([4,100], stddev=1, mean=0), name='W1', dtype=tf.float32)
W2 = tf.Variable(tf.random_normal([100,1], stddev=1, mean=0), name='W1', dtype=tf.float32)
B = tf.Variable(tf.random_normal([100], stddev=1, mean=0), name='B1', dtype=tf.float32)
x = tf.placeholder(tf.float32, [1,4])
WB = tf.matmul(tf.add(tf.matmul(x, W1), B), W2)
TR = tf.placeholder(tf.float32)
output = WB * 0 + totalreward

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(output)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i_episode in range(2000):
    observation = env.reset()
    totalreward = 0
    for t in range(2000):
        env.render()
        #print(observation)
        action = tf.squeeze(tf.squeeze(sess.run(WB, {x:[observation]} )))

        if action > 0 :
           action = 1
        else:
           action = 0
        #print(action)
        observation, reward, done, info = env.step(action)

        print(totalreward)
        print(action)

        totalreward -= reward
        sess.run(train,  {x:[observation], TR:totalreward})

        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            break
    #break


def setup_network():
   W1 = tf.Variable(tf.random_normal([4,10], stddev=1, mean=0), name='W1', dtype=tf.float32)
   W2 = tf.Variable(tf.random_normal([10,1], stddev=1, mean=0), name='W1', dtype=tf.float32)
   B = tf.Variable(tf.random_normal([10], stddev=1, mean=0), name='B1', dtype=tf.float32)
   x = tf.placeholder(tf.float32, [1,4])
   WB = tf.matmul(tf.add(tf.matmul(x, W1), B), W2)

   optimizer = tf.train.GradientDescentOptimizer(0.01)
   train = optimizer.minimize(WB)

   init = tf.global_variables_initializer()
   sess = tf.Session()
   sess.run(init)
   #print(sess.run([W,B]))
   #result = sess.run(WB,train, {x:[[1,5,3,4]]} )[0][0]
   #print(result)
   print(sess.run(WB, {x:[[1,5,3,4]]} ))


#setup_network()
