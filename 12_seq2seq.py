from classes.base import Seq2Seq

class BisayaSeqToSeq(Seq2Seq):
	def load_data(self):
		data = []
		data.append((list('what is your name'), list('unsa imong pangalan')))
		data.append((list('what is your gender'), list('lalaki ka o babae')))
		return data

def main():
	agent = BisayaSeqToSeq(epochs=200)
	model = agent.train()
	while True:
		out = agent.evaluate(list(input('Input: ')))
		print('Output: ',''.join(out))


main()