import sys
import glob
from classes.base import Seq2Seq
from keras.preprocessing.text import text_to_word_sequence


class BisayaSeqToSeq(Seq2Seq):
	def load_data(self):
		data = []
		paths = glob.glob('data/conversations/*.txt')
		length = len(paths)

		for p in paths:
			with open(p, "r") as file:
				lines = list(file)
				if 2 <= len(lines):
					for i in range(len(lines) - 1):
						client = lines[i]
						agent = lines[i+1]
						data.append((text_to_word_sequence(client), text_to_word_sequence(agent)))

		return data

def main():


	if 'train' == sys.argv[1]:
		agent = BisayaSeqToSeq(epochs=100)
		model = agent.train()
		model.save('.models/chatbot.h5')
	elif 'play' == sys.argv[1]:
		agent = BisayaSeqToSeq(model_filename='.models/chatbot.h5')


	while True:
		out, unk = agent.evaluate(text_to_word_sequence(input('<You>: ')))
		if 0 == len(unk):
			print('<Bot>: '+' '.join(out))
		else:
			print('<Bot>: what is ' + ' '.join(unk) + '?')

main()