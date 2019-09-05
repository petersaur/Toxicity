from flask import Flask, request, jsonify


import os
import gc
import re
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from collections import defaultdict

# Machine learning models
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model
from keras import backend as K

# Gensim (Word2Vec)
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

# NLTK 
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize


# ## Read in CSV files
#     - Training set data
#     - Testing set data
#     - Submission set data

# In[8]:
# Load embedding with pre-made Word2Vec model
w2v = gensim.models.KeyedVectors.load_word2vec_format('resources/GoogleNews-vectors-negative300-SLIM.bin.gz', 
                                                      binary = True)


contraction_list = {"ur": "you are", "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
# Contraction function to replace contractions
def contractions(text):
    
    # Find special characters that could be used as apostrophes 
    special = ["’", "‘", "´", "`"]
    
    # Replace these 'apostrophes' with an actual apostrophe
    for s in special:
        text = text.replace(s, "'")
    
    # Replace contraction with words in our contraction list
    words = [contraction_list[word] if word in contraction_list else word for word in text.split(" ")]
    return ' '.join(words)

chars = "/~!@#$%^&*)('.,][_+=?><\:;|}{" + '""'

# Create a special characters function to remove any special characters
def sp_char(text):
    for c in chars:
        text = text.replace(c, f' {c} ')
    return text

# Create function to clean the testing dataset
def cleaning(text):
    
    # Lowercase the text
    text = text.lower()
    
    # Replace the contractions
    text = contractions(text)
    
    # Remove any special characters
    text = sp_char(text)
    
    # Tokenize the words with word_tokenize from NLTK
    # tokens = word_tokenize(text) 
    
    # Return tokenized text
    return text

def build_embedding_matrix(word_index, total_vocab, embedding_size):
    matrix = np.zeros((total_vocab, embedding_size))
    for word, index in tqdm(word_index.items()):
        try:
            matrix[index] = w2v[word]
        except KeyError:
            pass
    return matrix



# We are limiting the training dataset to 100,000 rows due to time/computing constraints
train_df = pd.read_csv('resources/train.csv', nrows = 100000)
submission = pd.read_csv('resources/submission.csv')


train_df = train_df[['target', 'comment_text']]
train_df["comment_text"] = train_df["comment_text"].progress_apply(lambda text: cleaning(text))


test_df = pd.read_csv('resources/test.csv')
test_df["comment_text"] = test_df["comment_text"].progress_apply(lambda text: cleaning(text))


epochs=50
batch_size=128
max_words=100000
max_seq_size=256

transformer = Tokenizer(lower = True, filters='', num_words=max_words)
transformer.fit_on_texts(list(train_df["comment_text"].values) + list(test_df["comment_text"].values))

t_x = transformer.texts_to_sequences(train_df["comment_text"].values)
t_x = pad_sequences(t_x, maxlen = max_seq_size)


x_prediction = transformer.texts_to_sequences(test_df["comment_text"])
x_prediction = pad_sequences(x_prediction, maxlen = max_seq_size)


word_index = transformer.word_index
total_vocab = len(word_index) + 1
embedding_size = 300
w2v = build_embedding_matrix(transformer.word_index, total_vocab, embedding_size)


y = (train_df['target'].values > 0.5).astype(int)
X_train, X_test, y_train, y_test = train_test_split(t_x, y, random_state=6)


# In[ ]:





# In[28]:


model = keras.Sequential()
model.add(keras.layers.Embedding(total_vocab, embedding_size, weights=[w2v], trainable=False))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=3)



score = model.evaluate(X_test, y_test, batch_size=batch_size)

model.save('test.h5')
print(score)


graph = K.get_session().graph

# instantiate flask 
app = Flask(__name__)

# load the model

    
@app.route('/predict', methods=['GET','POST'])
def predict():
    with graph.as_default():
        s = request.args.get('s')
        score = model.predict([transformer.texts_to_sequences([s])])[0][0]
    return jsonify({'score': score})

if __name__ == '__main__':
    app.run(debug=True)



