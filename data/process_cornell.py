import pandas as pd
from ast import literal_eval
import re
import pickle

DELIM = ' \+\+\+\$\+\+\+ '
# pull conversations data
with open('cornell/movie_conversations.txt','r') as file:
	c_df = pd.read_csv(file, sep=DELIM, engine='python', header=None, names=['u1','u2', 'm', 'l'])
with open('cornell/movie_lines.txt','r') as file:
	l_df = pd.read_csv(file, sep=DELIM, engine='python', index_col=1, header=None, names=['l', 'u', 'n','t'])

conversations = []
tokens = []
total = c_df.shape[0]
for idx,row in c_df.iterrows():
	line_list = literal_eval(row['l'])
	conv = []
	for line_id in line_list:
		lines = l_df.loc[l_df['l'] == line_id]
		for l_idx, line in lines.iterrows():
			if pd.isna(line['t']): continue
			text = line['t'].lower();
			text = re.sub(r'[^a-z ]', '', text)
			seq = text.split()
			conv.append(seq)
			tokens = tokens + [w for w in seq if w not in tokens]
	conversations.append(conv)

	print('Processing {:0.4f}%'.format(idx/total*100))

with open('cornell/data.pkl','wb') as file:
	pickle.dump((conversations, tokens), file)