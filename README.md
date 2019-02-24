# Overview
We implemented Recurrent Neural Network with Long Short Term Memory to generate poetries and we trained it on (Microsoft dataset).

# Content:
- Used methods
- Prepaire the training data
- Vectorize the training data
- RNN-LSTM generative model
- Some generated poetries

# Used methods
We used methods from two Arificial Intelligence branches: SkipGram model from Natural Language Processing to vectorize the training data and RNN with LSTM from Machine Learning as generative model.
<p><img width="414" alt="image" src="https://user-images.githubusercontent.com/26183913/53301787-5fad1680-3857-11e9-863a-225348dfffd2.png"></p>

# Prepaire the training data
Poetries are type of text and they are sequential data. Unilike images, because one image could be fed to a neural network at once, but one poetry is not able to be processed by a neural network at once, because each poetry has differente count of words, thus we defined a sequence length parameter to process a sequence of words at once. RNN-LSTM takes sequence of words as input to predict the next word that should come after this sequence as output. Therefore, we created X-Data as sequences of 3 words from each poetries in the training data and Y-Data containes the next words of each sequence in the X-Data. Like in the following table:
<p><img width="974" alt="image" src="https://user-images.githubusercontent.com/26183913/53301820-b0bd0a80-3857-11e9-81d0-e626fc45f1e2.png"></p>

# Vectorize the training data
Words are not the input of RNN-LSTM. We need first to convert them to embedding vectors of the same size if the input layer of the RNN-LSTM. We chose 512 dimension for each embedding vector. However, it's not enouph just to assign each word to a unique random vector, we should use SkipGram model to convert the words to embedding vectors
[link to skipgram paper]
Firstly this model need to be trained on the poetries training data. After finishing the training process, we use the pretrained SkipGram model to vectorize the words sequences before feeding them to the RNN-LSTM.

The following picture represents embedding vectors of some of the words in our training data, but they have been projected from 512 dimension to 2 dimension using TSNE() tool in order to visualize them in 2d space.
![image](https://user-images.githubusercontent.com/26183913/53301864-20cb9080-3858-11e9-9af7-20f073632779.png)

# RNN-LSTM generative model
Our implemented RNN-LSTM consistes of 3 layers: an input layer of size of 3 embedding vectors(3 vectorized words) as we defined the sequence length parameter of size 3, a hidden layer of 512 LSTM cells and a dense output layer of the vocab_size (vocab_size is the count of all unique words in the poetries).
## RNN-LSTM training process
Firstly we load a batch from the training data: X-Data and Y-Data, then we vectorize X-Data words using the pretrained SkipGram model as 512 embedding vectors and we vectorize each word in Y-Data as dense vector of vocab_size, because the RNN-LSTM model will be trained to predict the next word of the sequence as a dense vector. The following picture represents the full AI model that consists of SkipGram model and RNN-LSTM model:
<p><img width="846" alt="image" src="https://user-images.githubusercontent.com/26183913/53301909-98012480-3858-11e9-825e-1d7a7c457cec.png"></p>
## Use the pretrained RNN-LSTM model to generate poetries
In order to use the RNN-LSTM pretrained model to generate a poem, we need a seed of 3 words, then we vectorize those three words using the pretrained SkipGram model to feed them to the RNN-LSTM. RNN-LSTM will predict the next word, then we create a new sequence of 3 words by using the second and the third words of the last sequence and by adding the new predicted word to the sequence and so on.

# Some generated poetries
- For our lives and fates enjoy the beauty of the mallard paddling paddling on the lake overlook her irridescent shawl shimmering green like a silk kimono ignore her resplendent composure as she drifts in spendor like cleopatras barge disregard her breast chestnut hued

- Of carven stone the lord of silver fountains shall come into his own his crown shall be upholden his harp shall be restrung his halls shall echo golden to songs of yore re sung the woods shall wave on mountains and grass beneath

- Daytime surrenders evening comes and the grass grows soft and white and blue the very rainbow showers have turned to blossoms where they fell and sown the earth with flowers and calls to the original of the moon the little dog laughed

- quiet and slow at a tranquil and all that jazz of the morning see the night slur into dawn hear the slow great winds arise where tall trees flank the way and shoulder toward the sky or wade with the wind

- And light my heart was also and still i will fly to be a part i feel the beating of its heart now knows the meaning to fly free and soar the streams without a goal my being trembles of delight a treasure

- I zest nothing but the light is made of many dyes the air is all perfume there's crimson buds and white and blue the very rainbow showers have turned to blossoms where they fell and sown the earth with flowers

- One must have a mind of winter to regard the frost and the boughs of the pine trees crusted with snow and have been cold a long time to behold the junipers shagged with ice the spruces rough in the distant glitter of

- Crusted in snow and from the forlorn world his visage hide stealing unseen to west with this disgrace even so my sun one early morn did shine with all triumphant splendor on my brow i am the daughter of the elements the somehow













