#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
data = pd.read_csv('ner_dataset.csv',encoding='unicode_escape')
data.head()


# In[10]:


vocab = list(set(data['Word'].to_list()))
idx2tok = {idx:tok for  idx, tok in enumerate(vocab)}


# In[19]:


def get_dict_map(data,token_or_tag):
    idx2Token={}
    token2Idx={}
    
    if token_or_tag == 'token':
        vocab = list(set(data['Word'].to_list()))
    else:
        vocab = list(set(data['Tag'].to_list()))
    
    idx2Token = {idx:token for idx,token in enumerate(vocab)}
    token2Idx = {token:idx for idx,token in enumerate(vocab)}
    
    return idx2Token,token2Idx


# In[21]:


idx2Token, token2idx  = get_dict_map(data, 'token')
idx2Tag, tag2idx  = get_dict_map(data, 'tag')


# In[25]:


data['Word_idx'] = data['Word'].map(token2idx)
data['Tag_idx'] = data['Tag'].map(tag2idx)
data.head()


# In[28]:


# Fill na
data_fillna = data.fillna(method='ffill', axis=0)
# Groupby and collect columns
data_group = data_fillna.groupby(
['Sentence #'],as_index=False
)['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))
# Visualise data
data_group.head()


# In[53]:


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
def get_pad_tokens_tags(data_group, data):
    #get max token and tag length
    n_token = len(list(set(data['Word'].to_list())))
    n_tag = len(list(set(data['Tag'].to_list())))
    
    tokens = data_group['Word_idx'].tolist()
    maxlen = max([len(s) for s in tokens])
    pad_tokens = pad_sequences(tokens,maxlen=maxlen,dtype='int32',padding='post',value=n_token-1)
    
    tags = data_group['Tag_idx'].tolist()
    maxlen = max([len(s) for s in tags])
    pad_tags = pad_sequences(tags,maxlen=maxlen,dtype='int32',padding='post',value=tag2idx["O"])
    
    n_tags = len(tag2idx)
    pad_tags = [to_categorical(i,num_classes=n_tags) for i in pad_tags]
    
    return pad_tokens,pad_tags


# In[54]:


pad_tokens,pad_tags = get_pad_tokens_tags(data_group,data)


# In[56]:


from sklearn.model_selection import train_test_split

#Split train, test and validation set
tokens_, test_tokens, tags_, test_tags = train_test_split(pad_tokens, pad_tags, test_size=0.1, train_size=0.9, random_state=2020)
train_tokens, val_tokens, train_tags, val_tags = train_test_split(tokens_,tags_,test_size = 0.25,train_size =0.75, random_state=2020)


# In[57]:


import numpy as np
import tensorflow
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import plot_model


# In[60]:


def build_lstm_model():
    model = Sequential()
    input_dim = len(list(set(data['Word'].to_list())))+1
    output_dim = 64
    input_length = max([len(s) for s in data_group['Word_idx'].tolist()])
    n_tags = len(tag2idx)
    # Add Embedding layer
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))

    # Add bidirectional LSTM
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode = 'concat'))

    # Add LSTM
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

    # Add timeDistributed Layer
    model.add(TimeDistributed(Dense(n_tags, activation="relu")))

    #Optimiser 
    # adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model


# In[61]:


def train_model(X, y, model):
    loss = list()
    for i in range(25):
        # fit model for one epoch on this sequence
        hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss


# In[62]:


model = build_lstm_model()


# In[ ]:


results = pd.DataFrame()
results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model)

