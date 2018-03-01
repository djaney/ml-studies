import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical



PAD = 0
END = 1

EPOCHS = 50

ENC_SEQ_SIZE = 10
DEC_SEQ_SIZE = 3

LATENT_DIM = 256

enc_tokens = []
dec_tokens = []
enc_token_map = []
dec_token_map = []
enc_token_size = 0
dec_token_size = 0

def enc_idx_to_val( idx):
	return enc_tokens[idx - 1]
def enc_val_to_idx(val):
	return enc_token_map[val] + 1
def dec_idx_to_val(idx):
	return dec_tokens[idx - 2]
def dec_val_to_idx(val):
	return dec_token_map[val] + 2

def data_batch(inp, out):

	# x,y,z encoder input, decoder input, decoder target
	x = []
	y = []
	z = []

	enc = [enc_val_to_idx(i) for i in inp]
	end_target = [dec_val_to_idx(i) for i in out]
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
	
	x = to_categorical(x, num_classes=enc_token_size)
	y = to_categorical(y, num_classes=dec_token_size)
	z = to_categorical(z, num_classes=dec_token_size)

	return x,y,z

def load_data():
	data = []
	data.append(('what is your name'.split(' '), 'unsa imong pangalan'.split(' ')))
	data.append(('what is your gender'.split(' '), 'lalaki ka o babae'.split(' ')))
	load_tokens(data)
	return data

def load_tokens(data):
	global enc_tokens
	global dec_tokens
	global enc_token_map
	global dec_token_map
	global enc_token_size
	global dec_token_size
	for d in data:
		enc_tokens = enc_tokens + d[0]
		dec_tokens = dec_tokens + d[1]
		enc_tokens = list(set(enc_tokens))
		dec_tokens = list(set(dec_tokens))
	enc_token_map = dict((c, i) for i, c in enumerate(enc_tokens))
	dec_token_map = dict((c, i) for i, c in enumerate(dec_tokens))
	enc_token_size = len(enc_tokens) + 1
	dec_token_size = len(dec_tokens) + 2 # add 2 for pad and end

def data_all():
	data = load_data()
	x, y, z = data_batch(data[0][0],data[0][1])
	for d in data[1:]:
		_x, _y, _z = data_batch(d[0],d[1])
		x = np.concatenate((x,_x), axis=0)
		y = np.concatenate((y,_y), axis=0)
		z = np.concatenate((z,_z), axis=0)
	return x, y, z

def train():

	x,y,z = data_all()

	# Define an input sequence and process it.
	encoder_inputs = Input(shape=(None, enc_token_size))
	encoder = LSTM(LATENT_DIM, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, dec_token_size))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the 
	# return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(LATENT_DIM, return_sequences=False, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
	                                     initial_state=encoder_states)
	decoder_dense = Dense(dec_token_size, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)

	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

	

	# Run training
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	model.fit([x, y], z,
		epochs=EPOCHS)

	return model

def evaluate(model, inp):
	out = []

	enc = [enc_val_to_idx(i) for i in inp.split(' ')]
	dec = [PAD] * DEC_SEQ_SIZE
	enc = np.array([to_categorical(enc, num_classes=enc_token_size)])
	dec = np.array([to_categorical(dec, num_classes=dec_token_size)])

	# predict first
	x = [enc,dec]
	pred = model.predict(x)
	pred_class = np.argmax(pred[0])
	out.append(dec_idx_to_val(pred_class))

	for _ in range(100):
		dec = [[dec_val_to_idx(i) for i in out]]
		dec = pad_sequences(dec, maxlen=DEC_SEQ_SIZE, value=PAD)
		dec = np.array(to_categorical(dec, num_classes=dec_token_size))

		x = [enc,dec]
		pred = model.predict(x)
		pred_class = np.argmax(pred[0])
		if pred_class == END:
			break
		out.append(dec_idx_to_val(pred_class))

	return out

def main():
	model = train()
	while True:
		out = evaluate(model, input('Input: '))
		print('Output: ',' '.join(out))


main()