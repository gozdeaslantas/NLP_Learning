#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import nltk
from nltk import tokenize
import re
import gensim
import logging

#-------read & append paragraphs from file --------------------#
list = []
text = ""
for x in range(1,16):
    with open('./Corpus/abbasGuclu/'+ str(x) + '.txt', 'r') as f:
        text += f.read().replace('\n', " ")
    with open('./Corpus/emreKongar/'+ str(x) + '.txt', 'r') as f:
        text += f.read().replace('\n', " ")
    with open('./Corpus/fatihAltayli/'+ str(x) + '.txt', 'r') as f:
        text += f.read().replace('\n', " ")
    with open('./Corpus/hakkiDevrim/'+ str(x) + '.txt', 'r') as f:
        text += f.read().replace('\n', " ")
    with open('./Corpus/mahfiEgilmez/'+ str(x) + '.txt', 'r') as f:
        text += f.read().replace('\n', " ")


#text = "".join(str(e) for e in list)
#print(text)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#------preprocessing tasks -------------------------------------#

logging.info('Preprocessing has started')
text = text.lower()

nltk.download('punkt')
list = tokenize.sent_tokenize(text)

list_without_punct = [re.sub(r'[^\w\s]', "", x) for x in list]

clean_text = " ".join(list_without_punct)
logging.info('Preprocessing has finished')

print('unique word count: ', len(set(clean_text.split())))
print('whole word count: ', len(clean_text.split()))

logging.info('Tokenize words')
words = tokenize.word_tokenize(clean_text)

nltk.download('stopwords')
stop_word_list = nltk.corpus.stopwords.words('turkish')
filtered_words = [token for token in words if token not in stop_word_list]


logging.info('Stemming words')
from snowballstemmer import TurkishStemmer
turkStem=TurkishStemmer()
stemmed_clean_text = []
for w in filtered_words:
    stemmed_clean_text.append(turkStem.stemWord(w))


logging.info('Convert list into list of list for word2Vec')
list_of_list = [[x] for x in stemmed_clean_text]

#CBOW Model
logging.info('Cbow Model will be trained')
cbowModel = gensim.models.Word2Vec(list_of_list, size=100, window=2, min_count=1, workers=4, sg=0)
cbowModel.train(list_of_list,total_examples=len(list_of_list),epochs=10)
cbowModel.init_sims(replace=True)


#Find the similarities
cbowModel.wv.most_similar(positive=["ülke"])


#Find the similarities
cbowModel.wv.most_similar(positive=["genç"])


#Find the similarities
cbowModel.wv.most_similar(positive=["millet"])



# similarity between two different words
cbowModel.wv.similarity(w1="ağustos",w2="şubat")


#SKIP-GRAM Model
logging.info('Cbow Model will be trained')
skipGramModel = gensim.models.Word2Vec(list_of_list, size=100, window=2, min_count=1, workers=4, sg=1)
skipGramModel.train(list_of_list,total_examples=len(list_of_list),epochs=10)
skipGramModel.init_sims(replace=True)


# similarity between two different words
skipGramModel.wv.similarity(w1="ağustos",w2="şubat")


#Find the similarities
skipGramModel.wv.most_similar(positive=["ülke"])



#Find the similarities
skipGramModel.wv.most_similar(positive=["ağustos"])


#Find the similarities
cbowModel.wv.most_similar(positive=["ağustos"])
