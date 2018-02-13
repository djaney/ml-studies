import numpy as np
import glob
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Reshape
from keras.preprocessing import sequence

CHARMAP = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-=!@#$%^&*()_+`~[]\{}|;':\",./<>?"

SEQLEN = 5
BATCHSIZE = 10
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
    x = np.array(x)
    y = np.array(y)
    return x,y


def create_model():
    model = Sequential()
    model.add(SimpleRNN(SEQLEN*ALPHASIZE,input_shape=(SEQLEN, ALPHASIZE)))
    model.compile(optimizer='sgd',loss='binary_crossentropy')
    return model


model = create_model()

for i in range(1):
    file_data = get_file_data(FILES, i)
    idx = 0
    while True:
        x,y = build_line_data(file_data, SEQLEN, idx ,BATCHSIZE)
        model.fit(x, y, epochs=3, batch_size=BATCHSIZE)
        idx = idx + 1
        if 0 == len(x):
            break
        if idx > 10:
            break