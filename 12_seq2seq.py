import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

ENC_ALPHA = list(['what', 'is', 'your', 'name'])
DEC_ALPHA = list(['unsa', 'imong', 'pangalan'])
ENC_ALPHA_MAP = dict((c, i) for i, c in enumerate(ENC_ALPHA))
DEC_ALPHA_MAP = dict((c, i) for i, c in enumerate(DEC_ALPHA))

PAD = 0
END = 1

EPOCHS = 100
ENC_ALPHA_SIZE = len(ENC_ALPHA) + 1
DEC_ALPHA_SIZE = len(DEC_ALPHA) + 2 # add 2 for pad and end
ENC_SEQ_SIZE = 10
DEC_SEQ_SIZE = 3

LATENT_DIM = 256

def enc_idx_to_val( idx):
	return ENC_ALPHA[idx - 1]
def enc_val_to_idx(val):
	return ENC_ALPHA_MAP[val] + 1
def dec_idx_to_val(idx):
	return DEC_ALPHA[idx - 2]
def dec_val_to_idx(val):
	return DEC_ALPHA_MAP[val] + 2

def data_batch(inp, out):


	# x,y,z encoder input, decoder input, decoder target
	x = []
	y = []
	z = []

	enc = [enc_val_to_idx(i) for i in inp.split(' ')]
	end_target = [dec_val_to_idx(i) for i in out.split(' ')]
	end_target.insert(0, PAD)
	end_target.append(END)
	
	for i in range(len(end_target)-1):
		
		dec = end_target[0:i+1]
		tar = end_target[i+1]
		x.append(enc)
		y.append(dec)
		z.append(tar)
	x = np.array(pad_sequences(x, maxlen=ENC_SEQ_SIZE, value=PAD))
	y = np.array(pad_sequences(y, maxlen=DEC_SEQ_SIZE, value=PAD))
	x_shape = x.shape
	y_shape = y.shape
	
	x = to_categorical(x, num_classes=ENC_ALPHA_SIZE)
	y = to_categorical(y, num_classes=DEC_ALPHA_SIZE)
	z = to_categorical(z, num_classes=DEC_ALPHA_SIZE)

	return x,y,z

def train():
	# Define an input sequence and process it.
	encoder_inputs = Input(shape=(None, ENC_ALPHA_SIZE))
	encoder = LSTM(LATENT_DIM, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, DEC_ALPHA_SIZE))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the 
	# return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(LATENT_DIM, return_sequences=False, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
	                                     initial_state=encoder_states)
	decoder_dense = Dense(DEC_ALPHA_SIZE, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)

	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


	x,y,z = data_batch('what is your name', 'unsa imong pangalan')


	# Run training
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	model.fit([x, y], z,
		epochs=EPOCHS)

	return model

def evaluate(model, inp):
	out = []

	enc = [enc_val_to_idx(i) for i in inp.split(' ')]
	dec = [PAD] * DEC_SEQ_SIZE
	enc = np.array([to_categorical(enc, num_classes=ENC_ALPHA_SIZE)])
	dec = np.array([to_categorical(dec, num_classes=DEC_ALPHA_SIZE)])

	# predict first
	x = [enc,dec]
	pred = model.predict(x)
	pred_class = np.argmax(pred[0])
	out.append(dec_idx_to_val(pred_class))

	for _ in range(100):
		dec = [[dec_val_to_idx(i) for i in out]]
		dec = pad_sequences(dec, maxlen=DEC_SEQ_SIZE, value=PAD)
		dec = np.array(to_categorical(dec, num_classes=DEC_ALPHA_SIZE))

		x = [enc,dec]
		pred = model.predict(x)
		pred_class = np.argmax(pred[0])
		if pred_class == END:
			break
		out.append(dec_idx_to_val(pred_class))

	return ' '.join(out)

def main():
	model = train()
	while True:
		out = evaluate(model, input('Input: '))
		print('Output: ',out)


main()