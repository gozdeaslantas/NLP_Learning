#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import random

list = []
for x in range(1, 10):
    d = {}
    my_file = open('./Corpus/abbasGuclu/' + str(x) + '.txt')
    d['author'] = 'abbasGuclu'
    d['text'] = my_file.read()
    my_file.close()
    list.append(d)
    d = {}
    my_file = open('./Corpus/balcicekPamir/' + str(x) + '.txt')
    d['author'] = 'balcicekPamir'
    d['text'] = my_file.read()
    my_file.close()
    list.append(d)
    my_file = open('./Corpus/eceTemelkuran/' + str(x) + '.txt')
    d = {}
    d['author'] = 'eceTemelkuran'
    d['text'] = my_file.read()
    my_file.close()
    list.append(d)
    my_file = open('./Corpus/gulseBirsel/' + str(x) + '.txt')
    d = {}
    d['author'] = 'gulseBirsel'
    d['text'] = my_file.read()
    my_file.close()
    list.append(d)
    d = {}
    my_file = open('./Corpus/guneriCivaoglu/' + str(x) + '.txt')
    d['author'] = 'guneriCivaoglu'
    d['text'] = my_file.read()
    my_file.close()
    list.append(d)

random.shuffle(list)
df = pd.DataFrame(list)
df

#######  START PREPROCESS DATA ###########################################################################
import nltk
import re
import numpy as np

nltk.download('stopwords')
stop_word_list = nltk.corpus.stopwords.words('turkish')
WPT = nltk.WordPunctTokenizer()

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    tokens = WPT.tokenize(text)
    text = ' '.join(word for word in tokens if word not in stop_word_list) # delete stopwors from text
    return text
    
df['text'] = df['text'].apply(clean_text)


df['text'].apply(lambda x: len(x.split(' '))).sum()

print(df['text'])
#######  END PREPROCESS DATA ###########################################################################

df['author_num'] = df.author.map({'abbasGuclu':0, 'balcicekPamir':1, 'eceTemelkuran':2, 'gulseBirsel':3, 'guneriCivaoglu':4})

x = df['text']
y = df['author_num']


######  DATA SPLIT FOR TRAINING AND TESTING #############################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=435)


## LOGISTIC REGRESSION CLASSIFICATION #################################################################

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)

#########  METRICS ###############################################################################################
y_pred_test = logreg.predict(X_test)
authors = df['author'].unique().tolist()
#y_pred_test
print('accuracy %s' % accuracy_score(y_pred_test, y_test))
print(classification_report(y_test, y_pred_test,target_names=authors))
