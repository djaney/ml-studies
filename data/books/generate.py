# coding=utf8
import csv
import urllib2
import re
all_words = []
with open('books.csv', 'r') as csvfile:
	books = csv.reader(csvfile, delimiter=',', quotechar='"')
	for line in books:
		dictionary = {}
		filename = line[0]
		label = line[1]
		response = urllib2.urlopen(filename)
		book_string = response.read()

		regex = ur"(\w+)"

		test_str = book_string

		matches = re.finditer(regex, test_str, re.IGNORECASE)

		for matchNum, match in enumerate(matches):
		    for groupNum in range(0, len(match.groups())):
		    	word = match.group(1).lower()
		    	if word in dictionary:
		    		dictionary[word] = dictionary[word] + 1
		    	else:
		    		dictionary[word] = 1
				
				if word not in all_words:
					all_words.append(word)

all_words.sort()
print(all_words)