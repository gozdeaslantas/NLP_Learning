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


# In[231]:


df['author_num'] = df.author.map({'abbasGuclu':0, 'balcicekPamir':1, 'eceTemelkuran':2, 'gulseBirsel':3, 'guneriCivaoglu':4})


# In[232]:


x = df['text']
y = df['author_num']


# In[243]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=434)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[244]:


# examine the class distribution in y_train and y_test
print(y_train.value_counts(),'\n', y_test.value_counts())


# In[245]:


# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# vect = CountVectorizer()
# vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b')
vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\;|\:')
# vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\?|\;|\:|\!|\'')
vect


# In[246]:


# learn the vocabulary in the training data, then use it to create a document-term matrix
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix created from X_train
X_train_dtm


# In[247]:


# transform the test data using the earlier fitted vocabulary, into a document-term matrix
X_test_dtm = vect.transform(X_test)
# examine the document-term matrix from X_test
X_test_dtm


# In[248]:


def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[249]:


from string import punctuation
X_train_chars = X_train.str.len()
X_train_punc = X_train.apply(lambda x: len([c for c in str(x) if c in punctuation]))
X_test_chars = X_test.str.len()
X_test_punc = X_test.apply(lambda x: len([c for c in str(x) if c in punctuation]))
X_train_dtm = add_feature(X_train_dtm, [X_train_chars, X_train_punc])
X_test_dtm = add_feature(X_test_dtm, [X_test_chars, X_test_punc])


# In[250]:


# import and instantiate the Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
# set with recommended hyperparameters
nb = MultinomialNB(alpha=1.0)
# train the model using X_train_dtm & y_train
nb.fit(X_train_dtm, y_train)


# In[251]:


# make author (class) predictions for X_test_dtm
y_pred_test = nb.predict(X_test_dtm)


# In[252]:


# compute the accuracy of the predictions with y_test
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_test)


# In[253]:


# compute the accuracy of training data predictions
y_pred_train = nb.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)


# In[254]:


# look at the confusion matrix for y_test
metrics.confusion_matrix(y_test, y_pred_test)



