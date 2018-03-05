import sys
import pickle
from classes.base import Seq2Seq
from keras.preprocessing.text import text_to_word_sequence


class BisayaSeqToSeq(Seq2Seq):
def main():


	if 'train' == sys.argv[1]:
		data = []
		with open('data/chatbot/codecamp.pkl', 'rb') as file:
			lines = pickle.load(file)
		for i in range(len(lines)-1):
			client = lines[i]
			agent = lines[i+1]
			data.append((text_to_word_sequence(client.strip()), text_to_word_sequence(agent.strip())))
			if 0 == 500 % i:
				agent = BisayaSeqToSeq(epochs=10, data=data)
				model = agent.train()
				model.save('.models/chatbot.h5')
				data = []
	elif 'play' == sys.argv[1]:
		agent = BisayaSeqToSeq(model_filename='.models/chatbot.h5')


	while True:
		inp = input('<You>: ')
		out, unk = agent.evaluate(text_to_word_sequence(inp))
		if 0 == len(unk):
			print('<Bot>: '+' '.join(out))
		else:
			print('<Bot>: Sorry, I don\'t know ' + inp)

main()