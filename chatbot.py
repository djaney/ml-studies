import sys
import glob
from classes.base import Seq2Seq
from keras.preprocessing.text import text_to_word_sequence


class BisayaSeqToSeq(Seq2Seq):
	def load_data(self):
		data = []
		paths = glob.glob('data/chatbot/*.txt')
		for path in paths:
			with open(path) as file:
				lines = list(file);
				for i in range(len(lines)-1):
					client = lines[i]
					agent = lines[i+1]
					data.append((text_to_word_sequence(client.strip()), text_to_word_sequence(agent.strip())))

		return data

def main():


	if 'train' == sys.argv[1]:
		agent = BisayaSeqToSeq(epochs=100)
		model = agent.train()
		model.save('.models/chatbot.h5')
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