# Word2vec_feature
word2vec feature extraction 
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re

import nltk

import gensim.models.word2vec as w2v
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
      

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
corpus ='/Users/PHK/panchatantra/panchatantra1.txt'
corpusraw = u""
with codecs.open(corpus,"r","utf-8") as corpus:
      corpusraw +=corpus.read()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(corpusraw)

def sentence_to_wordlist(raw):
      clean = re.sub("[^a-zA-Z]"," ",raw)
      word = clean.split()
      return word

sentences = []
for raw_sentence in raw_sentences:
      if len(raw_sentence)>0:
            sentences.append(sentence_to_wordlist(raw_sentence))

print(raw_sentences[2])
print(sentence_to_wordlist(raw_sentences[2]))
token_count = sum([len(sentence) for sentence in sentences])
print("The vorpus contains{0:,} tokens".format(token_count))


numfeature = 400
minword_cn =3
num_wrks = multiprocessing.cpu_count()

context_window = 7
downsampling = 3
seed = 1 

wordvector = w2v.Word2Vec(sentences = None,seed = seed, workers = num_wrks,size= numfeature,min_count = minword_cn,window = context_window, sample = downsampling)
wordvector.build_vocab(sentences)
print("word2vec vocabulary length: ", len(wordvector.wv.vocab))
pretrained_weights = wordvector.wv.syn0
vocab_size, embedding_size = pretrained_weights.shape


#wordvector.train(sentences)


#wordvector = w2v.Word2Vec.load(os.path.join("trained","wordvector.w2v"))
tsne = TSNE(n_components = 2, random_state = 0 )
vocab = list(wordvector.wv.vocab)
X = wordvector[vocab]
X_tsne = tsne.fit_transform(X)

df = pd.concat([pd.DataFrame(X_tsne),
                pd.Series(vocab)],
               axis=1)

df.columns = ['x', 'y', 'word']
N=len(wordvector.wv.vocab)
colors = np.random.rand(N)
plt.scatter(df['x'],df['y'])
area = np.pi * (15 * np.random.rand(N))**2
plt.scatter(df['x'],df['y'], s= area, c= colors, alpha = 0.5)

for i, txt in enumerate(df['word']):
    plt.annotate(txt, (df['x'].iloc[i], df['y'].iloc[i]))

plt.show()

use_input = input('give me a word for similarity test: ')
print(wordvector.most_similar(use_input))
