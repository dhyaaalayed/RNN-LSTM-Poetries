import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import json
import re
import tensorflow as tf
from skip_gram import *


np.set_printoptions(threshold=np.nan)

json_reader = Json_reader()

poetries = json_reader.get_poetries()

tokenizer = text.Tokenizer()
word2id, id2word, poetries_from_keras, poetries_words_from_keras = json_reader.get_dictionaries_from_tokenizer(tokenizer)
vocab_size = len(word2id) + 1
print('word2id: ', word2id)
print('vocab_size: ', vocab_size)
dim_embedddings = 512
SkipGram = build_skip_gram(vocab_size, dim_embedddings)
### load the weights:
# SkipGram.load_weights('andrew_skipgram_e340.h5')
SkipGram.load_weights('embedding_weights_rms_512_e230.h5')
weights = SkipGram.get_weights()[0]
embedding_matrix = weights
print('after loading weights: ', weights.shape)

sequence_length = 3

sequences = []
targets = []
for poem in poetries:
	print('poem: ', poem)
	for i in range(0, len(poem) - sequence_length):
		current_sequence = [ [word] for word in poem[i:i+sequence_length] ]		
		current_target = [[poem[i+sequence_length]]]
		# print('current sequence: ', current_sequence, '___ target: ', current_target)
		sequences.append(current_sequence)
		targets.append(current_target)


numeric_sequences = []
numeric_targets = []
for poem in poetries:
	# print('poem: ', poem)
	for i in range(0, len(poem) - sequence_length):
		# current_sequence = [ [   word2id[poem[i]] - 1   ], [  word2id[poem[i+1]] -1  ], [   word2id[poem[i+2]] -1  ] ]		
		current_sequence = [ [word2id[word] - 1] for word in poem[i:i+sequence_length] ]		
		current_target = [[   word2id[poem[i+sequence_length]] -1   ]]
		#print('current sequence: ', current_sequence, '___ target: ', current_target)
		numeric_sequences.append(current_sequence)
		numeric_targets.append(current_target)



def int2dense(vocab_size, word_index):
	word_dense = np.zeros(vocab_size)
	word_dense[word_index] = 1
	return word_dense


embedding_sequences = []
embedding_targets = []
dense_targets = []
print('type of weights', type(weights))
counter = 0
# loop to convert Poems to X_Sequences [embeddings] and Y_targets [denses]
for poem in poetries:
	# print('poem: ', poem)
	counter += 1
	for i in range(0, len(poem) - sequence_length):
		#current_sequence = [   weights[word2id[poem[i]] - 1]    ,  weights[word2id[poem[i+1]] -1]   ,  weights[word2id[poem[i+2]] -1]  ]
		current_sequence = [ weights[word2id[word] - 1] for word in poem[i:i+sequence_length] ]
		current_dense_target = int2dense(vocab_size, word2id[poem[i+sequence_length]])
		# print('current sequence: ', current_sequence, '___ target: ', current_target)
		# print('current sequence: ', current_sequence, '___ current_dense_target: ', current_dense_target)
		embedding_sequences.append(current_sequence)
		dense_targets.append(current_dense_target)
		if counter % 1000 == 0:
			print('current epoch: ', counter)

embedding_sequences = np.array(embedding_sequences, dtype = float)

Data = embedding_sequences

target = dense_targets

data = np.array(Data, dtype = float)

print('data shape: ', data.shape)
print('target shape: ', len(target))

x_train = data
y_train = target

main = False

if main:

	model = build_rnn_lstm(jr.vocab_size, dim_embedddings = 512, sequence_length = sequence_length)	

	model.summary()
	history_loss = []
	batch_size = 10000
	nb_batches = data.shape[0] // batch_size
	for i in range(1000):
		# history = model.fit(x_train, y_train, epochs = 1000, validation_data = (x_test, y_test))
		for j in range(1, nb_batches):
			X = data[(j-1) * batch_size:j * batch_size]
			y = target[(j-1) * batch_size:j * batch_size]
			history_loss.append( model.train_on_batch(X, y) )
		#model.save_weights('lstm_weights_e' + str(i) + '.h5')
		#plt.plot(history.history['loss'])
		plt.plot(history_loss)
		plt.savefig('lstm_loss.png')
	results = model.predict(x_test)
		

	print('the results after training: ', results)
	print('the true data', y_test)
	
	