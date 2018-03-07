import sys
import pandas as pd
import pickle
from classes.base import Seq2Seq
from keras.preprocessing.text import text_to_word_sequence


def main():
	with open('data/chatbot/wordlist.txt') as file:
		tokens = []
	if 'train' == sys.argv[1]:
		data = []
		

		df = pd.read_csv('data/chatbot/chat.csv', usecols=['client','agent'])

		for idx, row in df.iterrows():
			client_text = text_to_word_sequence(row['client'])
			agent_text = text_to_word_sequence(row['agent'])
			tokens = tokens + client_text + agent_text
			data.append((client_text,agent_text))

		tokens = list(set(tokens))

		agent = Seq2Seq((tokens,tokens),epochs=50,internal_size=250)
		model = agent.train(data)
		data = []
		model.save('.models/chatbot.h5')

		with open('.models/chatbot_token.pkl','wb') as file:
			pickle.dump(tokens, file)


	elif 'play' == sys.argv[1]:
		with open('.models/chatbot_token.pkl','rb') as file:
			tokens = pickle.load(file)
		agent = Seq2Seq((tokens,tokens), model_filename='.models/chatbot.h5')


	while True:
		inp = input('<You>: ')
		out, unk = agent.evaluate(text_to_word_sequence(inp))
		if 0 == len(unk):
			print('<Bot>: '+' '.join(out))
		else:
			print('<Bot>: Sorry, I don\'t know ' + inp)

main()