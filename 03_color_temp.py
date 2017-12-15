from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


data = np.array([
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0]
])

labels = np.array([[0,1],[1,0]])

batch_size = len(data);
num_steps = 1000
num_features = 3;
num_classes = 2;
learning_rate = 0.3
x = tf.placeholder(tf.float32, shape=[batch_size,num_features])
y = tf.placeholder(tf.int32, shape=[batch_size,num_classes])

def cnn_model_fn(features, labels, mode):
    hidden_layer = tf.layers.dense(inputs=features['colors'], units=16, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=hidden_layer, units=num_classes, activation=tf.nn.relu) # 2 classifications
    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

model = tf.estimator.Estimator(cnn_model_fn)

input_fn = tf.estimator.inputs.numpy_input_fn( x={'colors': data}, y=labels,
batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
model.train(input_fn, steps=num_steps)


# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'colors': data}, y=lables,
batch_size=batch_size, shuffle=False)

# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])