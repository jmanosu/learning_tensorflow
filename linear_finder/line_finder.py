import tensorflow as tf
import numpy

M = tf.Variable([.3], tf.float32)
B = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = M * x + B

y = tf.placeholder(tf.float32)

error = y - linear_model

squared_deltas = tf.square(error)

loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run([M, B]))

for i in range(1000):
	sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([M, B]))
