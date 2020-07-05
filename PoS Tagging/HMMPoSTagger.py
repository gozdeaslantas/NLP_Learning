#import
import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from nltk import word_tokenize

#read data
def read_data():
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
    
    #split data

train_data = read_data()
#for sent in train_data[:2]:
#    for tuple in sent:
#        print(tuple)

train_set, test_set = train_test_split(train_data, train_size=0.80, test_size=0.20, random_state=101)
train_tagged_words = [ tup for sent in train_set for tup in sent]
test_tagged_words = [ tup for sent in test_set for tup in sent]

#train1
def enumerate_list(data):
    return {instance: index for index, instance in enumerate(data)}

def calculateInitialTransitionP():
    tag_list = list(set([pair[1] for pair in train_tagged_words]))
    tag_id_list = enumerate_list(tag_list)  
    initial_probs = np.zeros(len(tag_list))
    transition_probs = np.zeros((len(tag_list),len(tag_list)))
    for seq in train_data:
        prev = 'None'
        for word, tag in seq:
            tag_id = tag_id_list[tag]
            if prev != 'None':
                    transition_probs[tag_id_list[prev], tag_id] += 1  
                    prev = tag
            else:
                    prev = tag
                    initial_probs[tag_id] += 1
    transition_probs = normalize(transition_probs)
    initial_probs = initial_probs/initial_probs.sum()
    return (initial_probs, transition_probs)
    
    #train2
def calculateEmissionP():
    tags = sorted(list(set([tag for sequence in train_data for word, tag in sequence])))
    tag_id_list = enumerate_list(tag_list)
    tokens = list([word.lower() for sequence in train_data for word, tag in sequence]) 
    words = list(set(tokens))
    word_id_list = enumerate_list(words)
    numTags = len(tags)
    numWords = len(words)
    emission_probs = np.zeros((numTags, len(words)))
    for sequence in train_data:
        for word, tag in sequence:
            word_lower = word.lower()
            tag_id = tag_id_list[tag]
            word_id = word_id_list[word_lower]
            emission_probs[tag_id, word_id] += 1
    emission_probs = normalize(emission_probs)
    return (emission_probs)


#helper
def normalize(matrix):
    row_sums = matrix.sum(axis=1)
    return matrix / row_sums[:, np.newaxis]
    
    
    def viterbi(obs, tags, start_p, trans_p, emit_p):
    V = [{}]
    for tag in tags:
        V[0][tag] = {"prob": start_p[tag] * emit_p[tag_id_list[tag],0], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in tags:
            max_tr_prob = V[t-1][tags[0]]["prob"]*trans_p[tag_id_list[tags[0]],tag_id_list[st]]
            prev_st_selected = tags[0]
            for prev_st in tags[1:]:
                tr_prob = V[t-1][prev_st]["prob"]*trans_p[tag_id_list[prev_st],tag_id_list[st]]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                    
            max_prob = max_tr_prob * emit_p[tag_id_list[st], t]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
                    
    for line in dptable(V):
        print (line)
    
    opt = []
    max_prob = 0.0
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st
    
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print ('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)
        
#train start
initial_probs, transition_probs = calculateInitialTransitionP()

tag_list = list(set([pair[1] for pair in train_tagged_words]))

tags = sorted(list(set([tag for sequence in train_data for word, tag in sequence])))
tag_id_list = enumerate_list(tag_list)
tokens = list([word.lower() for sequence in train_data for word, tag in sequence]) 
words = list(set(tokens))
word_id_list = enumerate_list(words)  
emission_probs = calculateEmissionP()
for sequence in train_data:
    for word,tag in sequence:
        word_lower = word.lower()
        tag_id = tag_id_list[tag]
        word_id = word_id_list[word_lower]
        print(emission_probs[tag_id, word_id])

tokens = list([word.lower() for sequence in test_set for word, tag in sequence]) 
words = list(set(tokens))

tag_list = list(set([pair[1] for pair in train_tagged_words]))
tags = sorted(list(set([tag for sequence in train_data for word, tag in sequence])))
print(tags)
print(initial_probs)

start_p  = dict(zip(tags, initial_probs))
len(emission_probs)
numTags = len(tags)
numWords = len(words)

obs = list(set(words))
viterbi(obs, tags, start_p, transition_probs, emission_probs)
