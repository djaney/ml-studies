import csv
import sys
import re
import pickle
from keras.preprocessing.text import text_to_word_sequence
csv.field_size_limit(sys.maxsize)

array = []
tokens = []
def flush(messages):
	global array
	global tokens
	message = '\n'.join(messages)
	message = filter(message)
	array.append(message)
	tokens = tokens + text_to_word_sequence(message);
	tokens = list(set(tokens))

def filter(text):
	text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
	text = re.sub(r'@[A-Za-z0-9]+', '<clientname>', text, flags=re.MULTILINE)
	
	return text

with open('/home/dmabelin/freecodecamp_casual_chatroom.csv') as file:
	last_user = None
	messages = []
	lines = csv.reader(file)
	for line in lines:
		current_user = line[8]
		if current_user != last_user and 0 < len(messages):
			flush(messages)
			messages = []

		
		messages.append(line[22])
		last_user = current_user

	if 0 < len(messages):
		flush(messages)
		messages = []
with open('data/chatbot/codecamp.pkl', 'wb') as data_pkl:
	pickle.dump((array, tokens), data_pkl)