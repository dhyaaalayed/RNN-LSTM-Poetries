# Content:
	- Overview
	- used models
	- training data
# Overview:
We implemented a generative model to generate poetries. This model has been trained on multim_poem.json file [has been taken from microsoft project]. Some of the generated poems:
- Among shrapnel bursts dazed by death's incandescence of the world and all i ask is a windy day with the white clouds flying and the flung spray and the blown spume and the sea gulls crying

- For our lives and fates enjoy the beauty of the mallard paddling paddling on the lake overlook her irridescent shawl shimmering green like a silk kimono ignore her resplendent composure as she drifts in spendor like cleopatras barge disregard her breast chestnut hued

- Of carven stone the lord of silver fountains shall come into his own his crown shall be upholden his harp shall be restrung his halls shall echo golden to songs of yore re sung the woods shall wave on mountains and grass beneath

- Daytime surrenders evening comes and the grass grows soft and white and blue the very rainbow showers have turned to blossoms where they fell and sown the earth with flowers and calls to the original of the moon the little dog laughed

- quiet and slow at a tranquil and all that jazz of the morning see the night slur into dawn hear the slow great winds arise where tall trees flank the way and shoulder toward the sky or wade with the wind

- And light my heart was also and still i will fly to be a part i feel the beating of its heart now knows the meaning to fly free and soar the streams without a goal my being trembles of delight a treasure

- I zest nothing but the light is made of many dyes the air is all perfume there's crimson buds and white and blue the very rainbow showers have turned to blossoms where they fell and sown the earth with flowers

- One must have a mind of winter to regard the frost and the boughs of the pine trees crusted with snow and have been cold a long time to behold the junipers shagged with ice the spruces rough in the distant glitter of

- Crusted in snow and from the forlorn world his visage hide stealing unseen to west with this disgrace even so my sun one early morn did shine with all triumphant splendor on my brow i am the daughter of the elements the somehow

# Methods used
Methods used in this project belong to two AI branches, Natural Language Processing to vectorize the training data and Machine Learning particulatly Deep Learning, because we need a Neural Network as generative model.
[photos of two AI branches]

# Vectorize our training data
Poetries are kind of text, and they are sequential data unlike images, because we can process one image inside a neural network at once. All what we need to vectorize images is to resize them to make all of them of the same size of the input layer of the neural network and sometimes to convert them to grayscale, but we cannot process one poetry inside a neural network at once, because every poetry has a different count of words, thus we need to define a sequence length to process a sequence of words at once, thus we chose Recurrent Neural Network which is widely used for generating text. We use word level method for this purpose to predict word by word 

RNN takes a sequence of words of a length and predicts the next word, thus we created our training data from the poetries that we have as follows:
[photos of the training data table]

but before we feed words into RNN, we need to vectorize them, which mean that each word that we have should be converted to embedding vector.
It's not enough just to assign each word with a unique random vector, we should use a Natural Language Processing model called SkipGram to convert our word to embedding vectors. [paper link of skipgram]

our data looks like vectors as in the following picture:
[picture of skipgram vectors scatter plot (only some of the words)]

After vectorizing our words we build the new RNN-LSTM model to take 3 embedding vectors according to our sequence length as input and to give us the next word as a dense vector as an output. The architecture is in the following figure:
[The very complex picture]

# How to use the code:
some explanaion about loading skipgram weights and rnn weights
[python RNN_generate]

# How to train the model:
1- training data as json file
2- use skipgram to vectorize them
3- load the saved skipgram weights to train rnn



















