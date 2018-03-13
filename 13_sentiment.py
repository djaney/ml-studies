#!/usr/bin/python3
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
docs = [
	"very good",
	"good",
	"alright",
	"awesome",
	"wonderful",
	"terrible",
	"dissapointing",
	"horrifying",
	"disgusting",
	"very bad"
]

labels = [1,1,1,1,1,0,0,0,0,0]

vocab_size = 50
doc_size = 10

tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(docs)
sequences = tokenizer.texts_to_sequences(docs)
sequences = pad_sequences(sequences,maxlen=doc_size, padding='post')
sequences = np.array(sequences)


model = Sequential()
model.add(Embedding(vocab_size, 3, input_length=doc_size))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile('rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(sequences, labels, epochs=50)

test = tokenizer.texts_to_sequences(['very bad'])
test = pad_sequences(test,maxlen=doc_size, padding='post')

prediction = model.predict(test)
print(model.summary())
print(prediction)