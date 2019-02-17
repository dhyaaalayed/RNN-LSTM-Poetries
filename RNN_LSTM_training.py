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

# def poem2wordlist(poem):
# 	clean = re.sub("[^a-zA-Z']"," ", poem) # solving the ' by adding ' to this regular expression in order to not removing it using re!
# 	words = clean.split()
# 	return words


# def get_poetries(data):
# 	poetries = []
# 	for poem in data: ## to have all poems
# 		poem_wordlist = poem2wordlist(poem['poem'])
# 		poetries.append(np.array(poem_wordlist))
# 	return poetries

json_reader = Json_reader()

# with open('multim_poem.json') as poem_file:
#     data = json.load(poem_file)

poetries = json_reader.get_poetries()

tokenizer = text.Tokenizer()
word2id, id2word, poetries_from_keras, poetries_words_from_keras = json_reader.get_dictionaries_from_tokenizer(tokenizer)
vocab_size = len(word2id) + 1 
print('vocab_size: ', vocab_size)
dim_embedddings = 512
SkipGram = build_skip_gram(vocab_size, dim_embedddings)
### load the weights:
# SkipGram.load_weights('andrew_skipgram_e340.h5')
SkipGram.load_weights('embedding_weights_rms_512_e230.h5')
weights = SkipGram.get_weights()[0]
embedding_matrix = weights
print('after loading weights: ', weights.shape)

sequence_length = 5

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

# print('sequences: ', sequences)
# print('targets: ', targets)


### apply the dictionary on the data:

numeric_sequences = []
numeric_targets = []
for poem in poetries:
	# print('poem: ', poem)
	for i in range(0, len(poem) - 3):
		current_sequence = [ [   word2id[poem[i]] - 1   ], [  word2id[poem[i+1]] -1  ], [   word2id[poem[i+2]] -1  ] ]
		current_target = [[   word2id[poem[i+3]] -1   ]]
		#print('current sequence: ', current_sequence, '___ target: ', current_target)
		numeric_sequences.append(current_sequence)
		numeric_targets.append(current_target)

# print('numeric_sequences: ', numeric_sequences)
# print('numeric_targets: ', numeric_targets)


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
	for i in range(0, len(poem) - 3):
		current_sequence = [   weights[word2id[poem[i]] - 1]    ,  weights[word2id[poem[i+1]] -1]   ,  weights[word2id[poem[i+2]] -1]  ]
		current_target = weights[word2id[poem[i+3]] -1]
		current_dense_target = int2dense(vocab_size, word2id[poem[i+3]])
		# print('current sequence: ', current_sequence, '___ target: ', current_target)
		# print('current sequence: ', current_sequence, '___ current_dense_target: ', current_dense_target)
		embedding_sequences.append(current_sequence)
		embedding_targets.append(current_target)
		dense_targets.append(current_dense_target)
		if counter % 1000 == 0:
			print('current epoch: ', counter)

embedding_sequences = np.array(embedding_sequences, dtype = float)

# print('embedding_sequences: ', embedding_sequences[:10])
print('embedding_sequences shape: ', embedding_sequences[:10].shape)
# print('embedding_targets: ', embedding_targets)

# the last problem: not all words are able to be converted to numbers using word2id
	# solution:
		# create manuall word2id just for this unim poem using simple python code!!!!!!!!

































###########________________






# Data = numeric_sequences
Data = embedding_sequences
# we normalize the data by dividing it on the range
# target = np.array(numeric_targets).flatten()



# target = np.array(embedding_targets).flatten() 
# target = np.array(embedding_targets) # this was used
target = dense_targets

# print('data: ', Data)
# print('target: ', target)

# How to slice
# data = np.array(Data[:500], dtype = float)
data = np.array(Data, dtype = float)
# target = np.array(target[:500], dtype = float)


# data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

# data = np.array(Data, dtype = float)
# target = np.array(target, dtype = float)

print('data shape: ', data.shape)
print('target shape: ', len(target))



# just few amount of data!!!!!

# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 4)
x_train = data
y_train = target

main = False

if main:

	# model = Sequential()
	# model.add(LSTM((1), batch_input_shape = (None, 3, 1), return_sequences = False))
	# model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['accuracy'])
	# model.compile( # this is Google TPU Shakspear compilation!
 #      optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
 #      loss='sparse_categorical_crossentropy',
 #      metrics=['sparse_categorical_accuracy'])

	model = Sequential()
	#model.add( Embedding(vocab_size, dim_embedddings, weights=[embedding_matrix], trainable=False, input_length=sequence_length) ) # input_length is the sequence
	model.add( LSTM(dim_embedddings, batch_input_shape = (None, sequence_length, dim_embedddings), return_sequences = False) )
	model.add(Dropout(0.2)) # should be added later
	model.add(Dense(vocab_size, activation='softmax')) # not sure about vocab_size - 1
	model.compile(loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'])
	

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
		model.save_weights('lstm_weights_e' + str(i) + '.h5')
		#plt.plot(history.history['loss'])
		plt.plot(history_loss)
		plt.savefig('lstm_loss.png')
	results = model.predict(x_test)
	
	
	# plt.scatter(range(results.shape[0]), results, c = 'red')
	# plt.scatter(range(y_test.shape[0]), y_test, c = 'green')
	
	# plt.show()
		

	print('the results after training: ', results)
	print('the true data', y_test)
	
	
	
	
	### id2words
	
	
	# results = np.round(results * unique_word_list.shape[0])
	
	# x_test = np.round(results * unique_word_list.shape[0])
	
	
	# for i in range(results.shape[0]):
	# 	print(unnormalized_word2id[x_test[i][0]], ' ') ## complete converting the poem












