from skip_gram import *
from json_reader import *
from stored_2d_weights import *
from values_arrays import *
np.random.seed(13)

def get_value_vectors(value_words, all_2d_vectors):
	value_vectors = []
	for i in range(len(value_words)):
		if value_words[i] in labels:
			value_vectors.append(all_2d_vectors[labels.index(value_words[i])])
	return np.array(value_vectors)

def transfer_weights_to_2d(weights):
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=3)
	T = tsne.fit_transform(weights)


json_reader = Json_reader()

poetries = json_reader.get_poetries()

tokenizer = text.Tokenizer()
word2id, id2word, poetries_from_keras, poetries_words_from_keras = json_reader.get_dictionaries_from_tokenizer(tokenizer)


skip_gram = build_skip_gram(json_reader.vocab_size, 512)
skip_gram.load_weights('embedding_weights_rms_512_e230.h5')
word_vectors = skip_gram.get_weights()[0]

# labels = [word for word in word2id]
labels = [id2word[i] for i in range(1, len(id2word) + 1)]


T = np.array(weights_2d)

print('after transformation...')


print(word2id)



def write_2d_to_file(file_name, vectors):
	with open(file_name, 'w') as f:
	       for i in range(len(vectors)):
	               vector_string = '['
	               for j in range(len(vectors[0])):
	                       if j != len(vectors[0]) - 1:
	                               vector_string += str(vectors[i][j]) + ', '
	                       else:
	                               vector_string += str(vectors[i][j])
	               vector_string += '],\n'
	               f.write(vector_string)

plt.figure(figsize=(14, 8))
# plt.scatter(T[:, 0], T[:, 1], c='steelblue', edgecolors='k')
# print('after scatter plotting...')
# for label, x, y in zip(labels, T[:, 0], T[:, 1]):
#     plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

### for specific values:

sensuality_vectors = get_value_vectors(Sensuality, T)
sociability_vectors = get_value_vectors(Sociability, T)
nature_vectors = get_value_vectors(Nature, T)
# values_labels = [Sensuality, Sociability, Nature]

plt.scatter(sensuality_vectors[:, 0], sensuality_vectors[:, 1], c = 'red')
for Sensuality, x, y in zip(Sensuality, sensuality_vectors[:, 0], sensuality_vectors[:, 1]):
    if Sensuality in labels:
    	plt.annotate(Sensuality, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

plt.scatter(sociability_vectors[:, 0], sociability_vectors[:, 1], c = 'green')
for Sociability, x, y in zip(Sociability, sociability_vectors[:, 0], sociability_vectors[:, 1]):
	if Sociability in labels:
		plt.annotate(Sociability, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

plt.scatter(nature_vectors[:, 0], nature_vectors[:, 1], c = 'yellow')
for Nature, x, y in zip(Nature, nature_vectors[:, 0], nature_vectors[:, 1]):
	if Nature in labels:
		plt.annotate(Nature, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

###
plt.show()


















