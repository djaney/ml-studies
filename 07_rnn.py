import numpy as np
import glob
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
CHARMAP = " \nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-!()\",.?"

SEQLEN = 40
BATCHSIZE = 1000
ALPHASIZE = len(CHARMAP)
INTERNALSIZE = 512
FILES = "shakespeare/*.txt"
LEARNING_RATE = 0.001
EPOCHS = 10


## Data related stuff
        
def char_to_value(char):
    idx = CHARMAP.find(char)
    if idx >= 0:
        return idx
    else:
        return None

def char_to_class_map(char):
    value = char_to_value(char)
    if value is None: return None
    return to_categorical(value,ALPHASIZE)
    
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
                line_values = [char_to_class_map(l) for l in line]
                line_values = [l for l in line_values if l is not None]
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
    model.add(LSTM(INTERNALSIZE,input_shape=(SEQLEN, ALPHASIZE), return_sequences=True))
    model.add(Dense(ALPHASIZE))
    model.add(Activation('softmax'))
    #adam optimizer
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

model = create_model()
fileIndex = 0
for fileIndex in range(42):
    file_data = get_file_data(FILES, fileIndex)
    idx = 0
    while True:
        x,y = build_line_data(file_data, SEQLEN, idx ,BATCHSIZE)
        print('File #'+str(fileIndex+1)+' Batch #'+str(idx+1))
        if 0 == len(x):
            break
        model.fit(x, y, epochs=EPOCHS, batch_size=BATCHSIZE)
        idx = idx + 1
        model.save('.models/07_rnn.model')
    

    fileIndex=fileIndex+1