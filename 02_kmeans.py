import tensorflow as tf
points = tf.Variable(tf.random_normal([2, 100]))
centroids = tf.Variable(tf.random_normal([2, 3]))

model = tf.global_variables_initializer()


with tf.Session() as session:
    print(session.run(model))