import sys
from classes.base import Seq2Seq
from keras.preprocessing.text import text_to_word_sequence


class BisayaSeqToSeq(Seq2Seq):
	def load_data(self):
		data = []
		data.append((text_to_word_sequence('what is your name'), text_to_word_sequence('unsa imong pangalan')))
		data.append((text_to_word_sequence('what is your gender'), text_to_word_sequence('lalaki ka o babae')))
		return data

def main():


	if 'train' == sys.argv[1]:
		agent = BisayaSeqToSeq(epochs=200)
		model = agent.train()
		model.save('.models/chatbot.h5')
	elif 'play' == sys.argv[1]:
		agent = BisayaSeqToSeq(model_filename='.models/chatbot.h5')


	while True:
		out = agent.evaluate(text_to_word_sequence(input('Input: ')))
		print('Output: ',' '.join(out))

main()