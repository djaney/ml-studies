import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import time
import math
import numpy as np
import glob

CHARMAP = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-=!@#$%^&*()_+`~[]\{}|;':\",./<>?"

SEQLEN = 5
BATCHSIZE = 1
ALPHASIZE = len(CHARMAP)
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout
FILES = "shakespeare/*.txt"


## Data related stuff
        
def char_to_value(char):
    idx = CHARMAP.find(char)
    if idx >= 0:
        return idx
    else:
        return 0
    
def value_to_char(value):
    return CHARMAP[value]

# iterate every single file
def get_file_data(pattern, index):
    paths = glob.glob(pattern)
    length = len(paths)
    
    if index < length:
        data = []
        with open(paths[index], "r") as file:
            for line in file:
                line_values = [char_to_value(l) for l in line]
                data = data + list(line_values)
        return data
    else:
        return None

# get batch data in file
def build_line_data(file_data, seqlen, batch_index, batch_count):
    length = len(file_data)
    start = batch_index * batch_count
    end = start+seqlen
    x = []
    y = []
    while end+1 <= length and len(x) < batch_count:
        x_line = file_data[start:end]
        y_line = file_data[start+1:end+1]
        x.append(x_line)
        y.append(y_line)
        start = start + 1
        end = start + seqlen
    return x,y


## Main program


## create model
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

# using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
# dynamic_rnn infers SEQLEN from the size of the inputs Xo

# How to properly apply dropout in RNNs: see README.md
cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
# "naive dropout" implementation
dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
H = tf.identity(H, name='H')  # just to give it a name

Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Ylogits = layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)


# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# init
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

for i in range(1):
    file_data = get_file_data(FILES, i)
    idx = 0
    while True:
        batch = build_line_data(file_data, SEQLEN, idx ,BATCHSIZE)
        
        
        # log
#         if 0 == step % 50:
#             print(ostate)
        
        step = step + 1
        idx = idx + 1
        if None == batch:
            break
        if idx > 10:
            break