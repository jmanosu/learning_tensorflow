import tensorflow as tf

m = tf.Variable([.1], tf.float32)
b = tf.Variable([-.3], tf.float32)

x = tf.placeholder(tf.float32)

linear_model = m * x + b

y = tf.placeholder(tf.float32)

sqaured_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(sqaured_delta)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

print(sess.rn(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
