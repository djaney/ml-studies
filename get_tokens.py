import pandas as pd
import sys
import re
import pickle
from keras.preprocessing.text import Tokenizer, one_hot

df = pd.read_csv('/home/dmabelin/freecodecamp_casual_chatroom.csv', usecols=['fromUser.id','text'])
print('CSV Loaded {} rows'.format(df.shape[0]))

print('Get tokens')

texts = []
for idx, row in df.iterrows():
	if pd.isna(row['text']):
		continue
	texts.append(row['text'])
	print(row['text'])
	print('------------------------')
	if idx > 10:
		break

tok = Tokenizer()
# create word_index
tok.fit_on_texts(texts)
# texts to numeric
seq = tok.texts_to_sequences(['that sounds like torture'])
print(seq)
hot = one_hot(['that sounds like torture'], tok.word_counts)
print(hot)

# with open('data/chatbot/tokens.pkl', 'wb') as data_pkl:
# 	pickle.dump(array, data_pkl)