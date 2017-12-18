# coding=utf8
import csv
import urllib2
import re
import struct
import pickle

all_words = []
data = {'x': [], 'y': []}
print('initialize words')
with open('words.csv', 'r') as file:
	all_words = [l.strip() for l in file]
with open('books.csv', 'r') as csvfile:
	books = csv.reader(csvfile, delimiter=',', quotechar='"')
	for line in books:
		dictionary = {}
		filename = line[0]
		label = line[1]
		print('generate data from '+filename)
		response = urllib2.urlopen(filename)
		book_string = response.read()

		regex = ur"([a-zA-Z\']{4,})"

		test_str = book_string

		matches = re.finditer(regex, test_str, re.IGNORECASE)

		for matchNum, match in enumerate(matches):
		    for groupNum in range(0, len(match.groups())):
		    	word = match.group(1).lower()
		    	if word in all_words:
			    	if word in dictionary:
			    		dictionary[word] = dictionary[word] + 1
			    	else:
			    		dictionary[word] = 1

		final = []
		for w in all_words:
			if w in dictionary:
				final.append(dictionary[w])
			else:
				final.append(0)

		data['x'].append(final)
		data['y'].append(label)

print('storing to binary file')
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)