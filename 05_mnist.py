
'''
Use MNIST database
'''

# Imports
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()
mnist = input_data.read_data_sets(FLAGS.data_dir)



# input - None is for batch, 3 is for number of input per batch
x = tf.placeholder(tf.float32, [None,784])
m = tf.layers.dense(x, 10, tf.nn.softmax)

# initialize the variables defined above
init = tf.global_variables_initializer()



# labes or correct answers
y = tf.placeholder(tf.int64, [None])

# calculate the loss distance using cross entropy
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=m)


optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_data = {x: batch_xs, y: batch_ys}

	sess.run(train_step, feed_dict=train_data)
	if 0 == i % 100:
		correct_prediction = tf.equal(tf.argmax(m, 1), y)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print(sess.run(accuracy, feed_dict={x: mnist.test.images,y: mnist.test.labels}))