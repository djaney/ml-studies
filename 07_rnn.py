import numpy as np
import glob
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import random
import os
import pickle
CHARMAP = " \n\tabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-!:()\",.?"

SEQLEN = 100
BATCHSIZE = 1000
ALPHASIZE = len(CHARMAP)
INTERNALSIZE = 512
FILES = "shakespeare/*.txt"
LEARNING_RATE = 0.001
EPOCHS = 1
TRIAL_FILE = '.trials/07_rnn';
MODEL_FILE = '.models/07_rnn.h5';

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

def res_to_word(res):
    paragraph = '';
    for line in res:
        for letter in line:
            char = value_to_char(np.argmax(letter))
            paragraph = paragraph+char
    return paragraph

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
        y_line = file_data[end+1]
        x.append(x_line)
        y.append(y_line)
        start = start + 1
        end = start + seqlen
    x = np.array(x)
    y = np.array(y)
    return x,y


def create_model():
    if os.path.isfile(MODEL_FILE):
        model = load_model(MODEL_FILE)
    else:
        model = Sequential()
        model.add(LSTM(INTERNALSIZE,input_shape=(SEQLEN, ALPHASIZE), dropout=0.2))
        model.add(Dense(ALPHASIZE))
        model.add(Activation('softmax'))
        #adam optimizer
        optimizer = Adam(lr=LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def run_trial(batchNumber):
    model = load_model(MODEL_FILE)
    words = '\tLONG TIME AGO IN A GALAXY FAR FAR AWAY\n' # start with a capital letter
    for _ in range(10):
        res = np.array([[char_to_class_map(x) for x in words]])
        res = pad_sequences(res, maxlen=SEQLEN)
        new_res = model.predict(res)
        words = words + res_to_word(new_res)


    with open('{}.txt'.format(TRIAL_FILE),'w')  as file:
        file.write(words) 


model = create_model()

if os.path.isfile(MODEL_FILE+'.pkl'):
    with open(MODEL_FILE+'.pkl', 'rb') as f:
        fileIndex, idx, batchNumber = pickle.load(f)
    recovery = True
else:
    recovery = False

model.save(MODEL_FILE)


run_trial(0)


if not recovery:
    fileIndex = 0
    batchNumber = 1
for fileIndex in range(fileIndex, 42):
    file_data = get_file_data(FILES, fileIndex)
    if not recovery:
        idx = 0
    else:
        recovery = False
    while True:
        x,y = build_line_data(file_data, SEQLEN, idx ,BATCHSIZE)
        print('File #'+str(fileIndex+1)+' Batch #'+str(batchNumber+1))
        if 0 == len(x):
            break
        model.fit(x, y, epochs=EPOCHS, batch_size=BATCHSIZE)
        model.save(MODEL_FILE)
        with open(MODEL_FILE+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([fileIndex, idx, batchNumber], f)

        if 0 == batchNumber % 50:
            run_trial(batchNumber)

        idx = idx + 1
        batchNumber = batchNumber + 1
    

    fileIndex=fileIndex+1
if os.path.isfile(MODEL_FILE+'.pkl'):
    os.remove(MODEL_FILE+'.pkl')