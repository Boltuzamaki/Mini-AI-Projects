
# coding: utf-8

# # Importing libraries

# In[14]:


from contraction import CONTRACTION_MAP
import re
import pickle
import math
import re
import time
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import nltk
from contraction import CONTRACTION_MAP     # Its a py file contain expanded word of all short words like I'm
from bs4 import BeautifulSoup
from tweepy import Stream
from tweepy import StreamListener
import json
import re
import csv
import tweepy
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np


# # Data cleaning functions

# In[15]:


def remove_htmltags(text):                    # Remove HTML tags
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_accented_chars(text):             # Normalizing accented charaters like ü
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP): # Expanding short words iike I've --> I have
    from contraction import CONTRACTION_MAP
    import contraction
    import re
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)                                if contraction_mapping.get(match)                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_special_characters(text, remove_digits=False):              # Remove special characters
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def simple_stemmer(text):                                             # Stemming the words
    import nltk
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def simple_lemmatize(text):                                          # lammetizing the words
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer() 
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def remove_stopwords(text, is_lower_case=False):                     # Remove stopwords
    from nltk.corpus import stopwords
    from nltk.tokenize import WordPunctTokenizer
    tokenizer = WordPunctTokenizer()
    stopword_list =stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def remove_link(text):                                                   # Remove https
    text = re.sub(r'http\S+', '', text)
    return text
    
def remove_hash_attherate(text):                                         # Remove @ and # tags
    text = re.sub("#\w*", "",text)
    text = re.sub("@\w*", "",text)
    text = re.sub("\s+", " ", text)
    return text

# Compiling all text cleaning function

def noramalize_text(text,htmltags = True, accented_chars = True, contractions_exp = True,
                   text_lower_case = True,special_characters = True, stemmer_text = True, 
                   lemmatize_text = True, stopwords_remove = False, remove_hash = True, remove_linkadd = True):
    if htmltags:
        text = remove_htmltags(text)
        
    if accented_chars:
        text = remove_accented_chars(text)
        
    if contractions_exp:
        text = expand_contractions(text)
        
    if text_lower_case:
            text = text.lower()
    
    if remove_linkadd:
        text = remove_link(text)
    # remove extra line
    text = re.sub(r'[\r|\n|\r\n]+', ' ',text)
        
    if remove_hash:
        text = remove_hash_attherate(text)
            
    if special_characters:
        text = remove_special_characters(text)
            
    if stemmer_text:
        text = simple_stemmer(text)
        
    if lemmatize_text:
        text = simple_lemmatize(text)
        
    # remove extra whitespace
    text = re.sub(' +', ' ', text)   
        
    if stopwords_remove:
        text = remove_stopwords(text) 
        
    return text


# # Loading the pretrained tokenizer and model 

# In[16]:


# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
MAX_LEN = 50    

from keras.models import load_model
mod = load_model('model.h5')


# In[17]:


sequences_test = tokenizer.texts_to_sequences(['This is good'])
test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test,value = 0,padding = 'post', maxlen = MAX_LEN)
pred = mod.predict(test)


# # Get it from twitter developer dashboard

# In[18]:


# inputs
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""


# In[20]:


sentiment = 0


# # Creating a function to write csv file of results to use it in plotting graph

# In[21]:


def csv_creator(sentiment_list):
    dictionary = { "sentiment" : sentiment_list
        }
    data = pd.DataFrame(dictionary, index = None)
    data.to_csv("real_time.csv", index = None)

import time    


# # Getting the tweets and predicting function

# In[22]:


text = []
class Listener(StreamListener):
    def __init__(self):
        self.sentiment = 0
        self.list = []
    def on_data(self, data):
        raw_tweets = json.loads(data)
        try:
            if  not raw_tweets['text'].startswith('RT'):              # "RT" to remove retweets
                text.append(noramalize_text(raw_tweets['text']))
                sequences_test = tokenizer.texts_to_sequences(text)
                test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test,value = 0,padding = 'post', maxlen = MAX_LEN)
                pred = mod.predict(test)
                if pred < 0.5:
                    self.sentiment = self.sentiment - 1
                if pred >= 0.5:
                    self.sentiment = self.sentiment + 1
                self.list.append(self.sentiment)  
                csv_creator(self.list)                      # Passing predicted list to csv_creator function
                time.sleep(2)
                print(self.sentiment)
                print(noramalize_text(raw_tweets['text']))
                text.pop()
                
        except:
            print("Error got")
    def on_error(self, status):
        print(status)


# # Put your authentication details here 

# In[23]:


auth = tweepy.OAuthHandler(consumer_key ,consumer_secret)
auth.set_access_token(access_token, access_token_secret)


# # Start real time tweet collecting steam 

# In[ ]:


twitter_stream = Stream(auth, Listener())
twitter_stream.filter(languages = ["en"], track = ['China'])

