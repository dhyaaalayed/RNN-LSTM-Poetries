import numpy as np
np.random.seed(13)

from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text

import gensim
from matplotlib import pyplot as plt
from json_reader import Json_reader

np.set_printoptions(threshold=np.nan)
 


def build_skip_gram(vocab_size, dim_embedddings):
	# inputs
	w_inputs = Input(shape=(1, ), dtype='int32')
	w = Embedding(vocab_size, dim_embedddings)(w_inputs)
	
	# context
	c_inputs = Input(shape=(1, ), dtype='int32')
	c  = Embedding(vocab_size, dim_embedddings)(c_inputs)
	o = Dot(axes=2)([w, c])
	o = Reshape((1,), input_shape=(1, 1))(o)
	o = Activation('sigmoid')(o)
	
	SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
	SkipGram.summary()
	SkipGram.compile(loss='binary_crossentropy', optimizer='adam')
	return SkipGram


def train_the_model(tokenizer, SkipGram, poetries, vocab_size):
	for epoch in range(0): # number of training steps
	    loss = 0.
	    for i, doc in enumerate(tokenizer.texts_to_sequences(poetries)): # check out later what this does exactly!!!!!!!!
	        data, labels = skipgrams(sequence=doc, vocabulary_size=vocab_size, window_size=5, negative_samples=5.)
	        x = [np.array(x) for x in zip(*data)]
	        y = np.array(labels, dtype=np.int32)
	        if x:
	            loss += SkipGram.train_on_batch(x, y)
	    print('epoch :', epoch,' loss: ' , loss) # need to be plotted later
	    if epoch % 10 == 0:
	    	# saving the wrights
	    	SkipGram.save_weights('andrew_skipgram_e' + str(epoch) + '.h5')
	    	# poltting the loss
	    	plt.plot(loss)
	    	plt.savefig('skip_gram_loss.png')

	weights = SkipGram.get_weights()[0] # embedding vectors
	return weights


main = True
if main == True:
	tokenizer = text.Tokenizer()
	jr = Json_reader()
	#word2id, id2word, poetries, poetries_words = get_dictionaries_from_tokenizer(tokenizer)
	word2id, id2word, poetries, poetries_words = jr.get_dictionaries_from_tokenizer(tokenizer = tokenizer)
	# print(word2id)
	print('Vocabulary Sample:', list(word2id.items())[:10])	
	vocab_size = jr.vocab_size
	print('vocab_size: ', vocab_size)
	dim_embedddings = 512
	SkipGram = build_skip_gram(vocab_size, dim_embedddings)
	
	### How to train the model	
	# weights = train_the_model(tokenizer, SkipGram, poetries, vocab_size)
	### End of How to train the model
	






	### How to get the embedding vector of a word:

	# embedding_vector = weights[word2id['or'] - 1]
	# print('embedding vector of or: ', embedding_vector)

	### End How to get the embedding vector of a word:
	
	














	# from sklearn.metrics.pairwise import euclidean_distances
	
	# distance_matrix = euclidean_distances(weights)
	# print(distance_matrix.shape)
	
	# similar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:6]+1] 
	#                    for search_term in ['god', 'sun', 'sky', 'tree', 'winter']} # not every word is exist!!!!!!!!!!!!!!
	
	# print(similar_words)































