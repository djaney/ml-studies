import sys
import pandas as pd
import pickle
from classes.base import Seq2Seq
from keras.preprocessing.text import text_to_word_sequence


def main():
	with open('data/chatbot/wordlist.txt') as file:
		tokens = text_to_word_sequence(file.read(), split=',')
	if 'train' == sys.argv[1]:
		data = []
		agent = Seq2Seq((tokens,tokens),epochs=50)

		df = pd.read_csv('/home/dmabelin/freecodecamp_casual_chatroom.csv', usecols=['fromUser.id','text'])

		# data.append((text_to_word_sequence('hello'), text_to_word_sequence('hi')))
		# model = agent.train(data)

		conv = []
		for idx, row in df.iterrows():
			if pd.isna(row['text']): continue
			conv.append(row['text'])
			if len(conv) > 1:
				data.append((text_to_word_sequence(conv[0]),text_to_word_sequence(conv[1])))
				conv.pop(0)
			if 99 == idx % 100:
				model = agent.train(data)
				data = []
				model.save('.models/chatbot.h5')
				print('{} of {}'.format(idx,df.shape[0]))
		if len(data) > 0:
			model = agent.train(data)
			data = []
			model.save('.models/chatbot.h5')
			print('{} of {}'.format(idx,df.shape[0]))

	elif 'play' == sys.argv[1]:
		agent = Seq2Seq((tokens,tokens), model_filename='.models/chatbot.h5')


	while True:
		inp = input('<You>: ')
		out, unk = agent.evaluate(text_to_word_sequence(inp))
		if 0 == len(unk):
			print('<Bot>: '+' '.join(out))
		else:
			print('<Bot>: Sorry, I don\'t know ' + inp)

main()