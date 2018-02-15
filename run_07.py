import numpy as np
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import sys
from keras.preprocessing.sequence import pad_sequences

CHARMAP = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-=!@#$%^&*()_+`~[]\{}|;':\",./<>?"
ALPHASIZE = len(CHARMAP)


## Data related stuff
        
def char_to_value(char):
    idx = CHARMAP.find(char)
    if idx >= 0:
        return idx
    else:
        return 0

def char_to_class_map(char):
    value = char_to_value(char)
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


def generate_random():
    pass

model = load_model('.models/07_rnn.model')
start = 'Long time ago, in a galaxy far far away,'
res = np.array([[char_to_class_map(x) for x in start]])
sys.stdout.write(start)
for _ in range(100):
    res = pad_sequences(res, maxlen=40)
    new_res = model.predict(res)
    words = res_to_word(new_res)
    sys.stdout.write(words)
    res = new_res