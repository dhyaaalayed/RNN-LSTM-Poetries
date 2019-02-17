from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import LSTM



def build_rnn_lstm(vocab_size, dim_embedddings, sequence_length):
	model = Sequential()
	#model.add( Embedding(vocab_size, dim_embedddings, weights=[embedding_matrix], trainable=False, input_length=sequence_length) ) # input_length is the sequence
	model.add( LSTM(dim_embedddings, batch_input_shape = (None, sequence_length, dim_embedddings), return_sequences = False) )
	model.add(Dropout(0.2)) # should be added later
	model.add(Dense(vocab_size, activation='softmax')) # not sure about vocab_size - 1
	model.compile(loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'])
	return model