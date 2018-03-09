import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt  
from classes.base import Seq2Seq
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import plot_model

def main():
	if 'train' == sys.argv[1]:
		model, history = train()
		if len(sys.argv) == 3 and sys.argv[2] == 'plot':
			print('Plotting chart')
			plot_model(model, to_file='.graphs/chatbot.png')
			plot(history)
	elif 'play' == sys.argv[1]:
		play()

def play():
	with open('.models/chatbot_token.pkl','rb') as file:
		tokens = pickle.load(file)
	agent = Seq2Seq((tokens,tokens), model_filename='.models/chatbot.h5')

	out, unk = agent.predict([])
	print('<Bot>: '+' '.join(out))
	while True:
		inp = input('<You>: ')
		out, unk = agent.predict(text_to_word_sequence(inp))
		if 0 == len(unk):
			print('<Bot>: '+' '.join(out))
		else:
			print('<Bot>: Sorry, I don\'t know ' + inp)

def train():
	data = []
	
	tokens = []
	df = pd.read_csv('data/chatbot/chat.csv', usecols=['client','agent'])

	for idx, row in df.iterrows():
		client_text = text_to_word_sequence(row['client']) if not pd.isna(row['client']) else []
		agent_text = text_to_word_sequence(row['agent'])
		tokens = tokens + client_text + agent_text
		data.append((client_text,agent_text))

	tokens = list(set(tokens))

	agent = Seq2Seq((tokens,tokens),internal_size=256,epochs=125)
	history = agent.train(data)
	model = agent.model
	data = []
	model.save('.models/chatbot.h5')

	with open('.models/chatbot_token.pkl','wb') as file:
		pickle.dump(tokens, file)

	out = ''
	inp = None

	while True:

		inp = input('<You>: ')

		if inp == '':
			break
		new_client = ' '.join(out);
		new_agent = inp
		res = df.query('client=="{}" & agent=="{}"'.format(new_client, new_agent))
		df2 = pd.DataFrame(data={'a': [new_client], 'b': [new_agent]})
		df.append({'client': [new_client], 'agent': [new_agent]}, ignore_index=True)
		with open('data/chatbot/chat.csv', 'a') as f:
			df2.to_csv(f, header=False, index=False)
			

		out, unk = agent.predict(text_to_word_sequence(inp))
		if 0 == len(unk):
			print('<Bot>: '+' '.join(out))
		else:
			print('<Bot>: Sorry, I don\'t know ' + inp)

	return (model, history)


def plot(history):
	plt.figure(1)  

	# summarize history for accuracy  

	plt.subplot(211)  
	plt.plot(history.history['acc'])  
	plt.plot(history.history['val_acc'])  
	plt.title('model accuracy')  
	plt.ylabel('accuracy')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'test'], loc='upper left')  

	# summarize history for loss  

	plt.subplot(212)  
	plt.plot(history.history['loss'])  
	plt.plot(history.history['val_loss'])  
	plt.title('model loss')  
	plt.ylabel('loss')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'test'], loc='upper left')  
	plt.show() 

main()