import numpy as np
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import sys

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




model = load_model('.models/07_rnn.model')
res = np.array([[char_to_class_map(x) for x in 'Long time ago, in a galaxy far far away']])
while True:
    res = model.predict(res)
    words = res_to_word(res)
    sys.stdout.write(words)