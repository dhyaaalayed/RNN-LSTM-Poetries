import numpy as np
np.random.seed(13) # maybe to get the same dictionary order every run
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
import json
import re


class Json_reader:

	def __init__(self):
		self.vocab_size = None # to share vocab size with all files
		self.word2id = None
		self.id2word = None
		self.poetries = None
		# self.poems_directory = 'our_datasets/all_andrew_data.json'
		self.poems_directory = 'multim_poem.json'
	def get_poetries(self): # private method
		with open(self.poems_directory) as poem_file:
		    data = json.load(poem_file)
		poetries = []
		for poem in data: ## to have all poems
			poem_wordlist = self.poem2wordlist(poem['poem'])			
			poetries.append(np.array(poem_wordlist))
		return poetries
	def poem2wordlist(self, poem): # private method
		# clean = re.sub('(\n)', '<eos>', poem)
		# We remove this line because the pre trained model has not been trained on eos!
		clean = poem
		clean = re.sub("[^a-zA-Z']", " ", clean) # solving the ' by adding ' to this regular expression in order to not removing it using re!
		clean = clean.lower() # to lower case: convert to small letters
		words = clean.split()
		#print(words)
		if words[0] == 'eos':
			words = words[1:] # delete the new line at the beginning if it exist
		return words
	def words2phrase(self, words): # private method
		phrase = ''
		for i in range(len(words)):
			if i == 0 and words[i] == 'eos': # ignore the first new line
				continue
			phrase += words[i] + ' '
		return phrase

	def get_dictionaries_from_tokenizer(self, tokenizer):
		with open(self.poems_directory) as poem_file:
			data = json.load(poem_file)
		poetries = []
		poetries_words = []
		for poem in data:
			words = self.poem2wordlist(poem['poem'])
			poetries_words.append(words)  
			phrase = self.words2phrase(words)
			poetries.append(phrase)
		print('all poems: ', len(poetries))
		tokenizer.fit_on_texts(poetries)
		print('end fitting the texts')
		word2id = tokenizer.word_index
		print('end word2id')
		id2word = {v:k for k, v in word2id.items()}
		print('end id2word')
		self.word2id = word2id
		self.id2word = id2word
		self.poetries = poetries
		self.poetries_words = poetries_words
		self.vocab_size = len(word2id) + 1
		return word2id, id2word, poetries, poetries_words
