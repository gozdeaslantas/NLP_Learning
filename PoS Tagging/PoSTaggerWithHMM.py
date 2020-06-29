#!/usr/bin/env python
# coding: utf-8

# In[19]:


#import libs
import nltk
from nltk import word_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('universal_tagset')


# In[12]:


text=word_tokenize("Aniden şiddetli bir yağmur başladı.")
nltk.pos_tag(text)


# In[17]:


nltk.corpus.brown.tagged_words()


# In[20]:


nltk.corpus.brown.tagged_words(tagset='universal')


# In[35]:


def read_data():
    """Read sentences from given corpus data"""
    all_sentences = []
    with open('POS_Tagged_Trained_Data.txt', 'r', encoding="utf8") as infile:
        sentence = []
        for line in infile:
            line = str.split(str.strip(line), '\t')
            if len(line) == 3:
                token, tag_label= line[0], line[2]
                sentence.append((token, tag_label))
                continue
            all_sentences.append(sentence)
            sentence = []
    return all_sentences


# In[174]:


import numpy as np
numOfTags = len(tags)  #states
numOfWords = len(words) #observations
initial_probs = np.zeros(numOfTags)
transition_probs= np.zeros((numOfTags, numOfTags)) #from state to state
emission_probs = np.zeros((numOfTags, numOfWords)) #which state which observation

tagStartCount = np.zeros(numOfTags)
tagTransitionCount= np.zeros((numOfTags, numOfTags)) #from state to state
emissionCount = np.zeros((numOfTags, numOfWords)) #which state which observation

train_size = len(all_sentences)
train_size = int(train_size * 50 / 100)
trainData, testData = all_sentences[:train_size], all_sentences[train_size:]
tags = sorted(list(set([tag for sequence in trainData for word, tag in sequence])))

def enumerate_list(data):
    return {instance: index for index, instance in enumerate(data)}

tag_dict = enumerate_list(tags)
words = list(set([word.lower() for sequence in trainData for word, tag in sequence]))
word_dict = enumerate_list(words)

def train():
    for sequence in trainData:
        prevTag = ''
        for word, tag in sequence:
            word = word.lower()
            if (prevTag != ''):
                tagTransitionCount[tag_dict[prevTag], tag_dict[tag]] += 1
            else:
                tagStartCount[tag_dict[tag]] += 1
            prevTag = tag

            wordId = word_dict[word]
            if  wordId not in emissionCount[tag_dict[tag]]:
                emissionCount[tag_dict[tag], word_dict[word]] = 1
            else:
                emissionCount[tag_dict[tag], word_dict[word]] += 1
    
    transition_probs = tagTransitionCount/tagTransitionCount.sum(1)[:, np.newaxis]
    initial_probs = tagStartCount/tagStartCount.sum()
    emission_probs = emissionCount/emissionCount.sum(1)[:, np.newaxis]
    return initial_probs, transition_probs, emission_probs


# In[ ]:


initial_probs, transition_probs, emission_probs = train()
print(transition_probs)
print(initial_probs)


# In[ ]:


def viterbi():

