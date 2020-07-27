#!/usr/bin/env python
# coding: utf-8

# In[2]:


def read_file(file_path):
    with open(file_path) as f:
        str_text = f.read()
        
    return str_text


# In[3]:


import spacy


# In[4]:


nlp = spacy.load('en', disable = ['parser', 'tagger', 'ner'])


# In[5]:


nlp.max_length=1198623


# In[6]:


def separate_punct(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']


# In[7]:


d = read_file('moby_dick_four_chapters.txt')
tokens = separate_punct(d)
tokens


# In[8]:


len(tokens)


# In[22]:


#pass in 25 words --> network predict #26


# In[12]:


train_len = 25 + 1
text_sequences = []

for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq) #input text_sequence of 26 token


# In[17]:


from keras.preprocessing.text import Tokenizer


# In[18]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)


# In[19]:


sequences = tokenizer.texts_to_sequences(text_sequences) #create unique indexes as id for each token in text_sequences


# In[24]:


#for i in sequences[0]:
#    print(f"{i} : {tokenizer.index_word[i]}")
#print(tokenizer.word_counts)


# In[28]:


vocab_size = len(tokenizer.word_counts) #unique token count


# In[31]:


import numpy as np
sequences = np.array(sequences)


# In[32]:


from keras.utils import to_categorical


# In[35]:


X = sequences[:,:-1]


# In[36]:


y = sequences[:,-1]


# In[37]:


y = to_categorical(y, num_classes=vocab_size+1)


# In[40]:


seq_len = X.shape[1]


# In[43]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding


# In[47]:


def create_model(vocab_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocab_size, seq_len, input_length=seq_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))
    
    model.add(Dense(vocab_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    model.summary()
    
    return model


# In[48]:


model = create_model(vocab_size+1, seq_len)


# In[49]:


from pickle import dump,load


# In[50]:


model.fit(X,y,batch_size=128,epochs=2,verbose=1)


# In[51]:


model.save('my_firstKeras_model.h5')
dump(tokenizer, open('dumped_tokenizer', 'wb'))


# In[52]:


############## text generation ############################################################


# In[53]:


from keras.preprocessing.sequence import pad_sequences


# In[70]:


def generate_text(model,tokenizer,seq_len,seed_text,num_gen_words):
    output_text = []
    
    input_text = seed_text
    for i in range(num_gen_words):
       # print(input_text)
       # print('----------------------')
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
       #  print(encoded_text)
       #  print('----------------------')
        pad_encoded = pad_sequences([encoded_text],maxlen=seq_len,truncating='pre')
        predicted_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        predicted_word = tokenizer.index_word[predicted_word_ind]
      #   print(predicted_word)
        input_text += ' '+predicted_word
        
        output_text.append(predicted_word)
    return ' '.join(output_text)


# In[55]:


import random
random.seed(101)
random_pick = random.randint(0,len(text_sequences))
random_seed_text = text_sequences[random_pick]
print(random_seed_text)


# In[60]:


seed_text = ' '.join(random_seed_text)
print(seed_text)


# In[66]:


generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=25)


# In[69]:


from keras.models import load_model


# In[71]:


model = load_model('epochBIG.h5')
tokenizer = load(open('epochBIG','rb'))


# In[72]:


generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=25)

