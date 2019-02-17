from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
import numpy as np
#from RNN_LSTM_training import *
from skip_gram import *
np.set_printoptions(threshold=np.nan)


vocab_size = 18638
dim_embedddings = 512
sequence_length = 3
model = Sequential()
model.add( LSTM(dim_embedddings, batch_input_shape = (None, sequence_length, dim_embedddings), return_sequences = False) )
model.add(Dropout(0.2)) # should be added later
model.add(Dense(vocab_size, activation='softmax')) # not sure about vocab_size - 1
model.compile(loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'])
model.summary()




# load weights:

model.load_weights('lstm_sequence_3/lstm_v2_weights_e130.h5')


## SkipGram:
SkipGram = build_skip_gram(vocab_size, dim_embedddings)

# SkipGram.load_weights('andrew_skipgram_e340.h5')
SkipGram.load_weights('embedding_weights_rms_512_e230.h5')
weights = SkipGram.get_weights()[0]


def embedding2id(embedding_matrix, embedding_vector):
	return np.where(embedding_matrix == embedding_vector)[0][0] + 1 # -1
def id2embedding(embedding_matrix, id):
	return embedding_matrix[id - 1]




def generate_next(seed_sequence):
	seed_sequence = np.expand_dims(seed_sequence, axis = 0)
	generated_dense = model.predict(seed_sequence)
	generated_id = np.argmax(generated_dense, axis = 1)[0]
	generated_embedding = id2embedding(weights, generated_id)
	next_sequence = np.array( [ seed_sequence[0][1], seed_sequence[0][2], generated_embedding  ] )
	return generated_id, generated_embedding, next_sequence

# next_id, next_embedding = generate_next(x_test[0])

def generate_poem_from_embedding(seed_sequence, nb_words):
	word_ids = []
	word_ids.append( embedding2id(weights, seed_sequence[0]) )
	word_ids.append( embedding2id(weights, seed_sequence[1]) )
	word_ids.append( embedding2id(weights, seed_sequence[2]) )
	next_sequence = seed_sequence
	print('first call shape: ', next_sequence.shape)
	for i in range(nb_words):
		next_id, next_embedding, next_sequence = generate_next(next_sequence)
		word_ids.append(next_id)

	generated_poem = []
	for i in range(len(word_ids)):
		generated_poem.append(id2word[word_ids[i]])

	return ' '.join(generated_poem)


def word_sequence2embedding_sequence(word_sequence):
	embedding_sequence = []
	embedding_sequence.append( id2embedding(weights, word2id[word_sequence[0]]))
	embedding_sequence.append( id2embedding(weights, word2id[word_sequence[1]]))
	embedding_sequence.append( id2embedding(weights, word2id[word_sequence[2]]))
	return np.array(embedding_sequence)


# for i in range(x_test.shape[0]):

# 	print('generated_poem: ', generate_poem(x_test[i], 40))
# print('generated_poem: ', generate_poem_from_embedding(x_test[0], 40))

word_sequence = np.array([ 'i', 'love', 'you' ])

embedding_sequence = word_sequence2embedding_sequence(word_sequence)

print(generate_poem_from_embedding(embedding_sequence, 40))



while True:
	input_sequence = input('please enter a seed of three words!\n')
	input_list = input_sequence.split()
	embedding_sequence = word_sequence2embedding_sequence(input_list)
	print(generate_poem_from_embedding(embedding_sequence, 40))	















