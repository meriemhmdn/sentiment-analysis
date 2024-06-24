import nltk                         # NLP toolbox
from os import getcwd
import pandas as pd                 # Library for Dataframes 
from nltk.corpus import twitter_samples 
import matplotlib.pyplot as plt     # Library for visualization
import numpy as np 
import pandas as pd
import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer
import os
from unidecode import unidecode
nltk.download('words')

from nltk.corpus import words
import csv
import json

stopwords_english = stopwords.words('english') 
words_to_remove = ["not"] #if you want to remove any stop word from the list
words_to_add = ["i'm"] #if you want to add any stop word to the list
    
for word in words_to_remove:
    if word in stopwords_english:
        stopwords_english.remove(word)
            
for word in words_to_add:
    if word in stopwords_english:
        stopwords_english.append(word)
        

def preprocess(text):
    
    text1 = unidecode(text.lower())
    
    text1 = re.sub(r'^RT[\s]+', '', text1)

    text1 = re.sub(r'https?://[^\s\n\r]+', '', text1)

    text1 = re.sub(r'#', '', text1)
    
    text1 = text1.replace("'", " ")
    
    text1 = text1.replace("_", " ")
    
    text1 = text1.replace("-", " ") 
    
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    text1_tokens = tokenizer.tokenize(text1)
    stopwords_english = stopwords.words('english') 
    alphabet = list(string.ascii_lowercase)
    
    
    text1_clean = []

    for word in text1_tokens: # Go through every word in your tokens list
        if((word not in stopwords_english) and (word not in string.punctuation) and (word not in alphabet)):  
            text1_clean.append(word)
    
    
    stemmer = PorterStemmer() 

    text1_stem = [] 
    
    for word in text1_clean:
        stem_word = stemmer.stem(word)
        text1_stem.append(stem_word) 
    
    
    text1_final = [] 
    for word in text1_stem: # Go through every word in your tokens list
        if((word not in stopwords_english) and (word not in string.punctuation) and (word not in alphabet)):  
            text1_final.append(word)
        
    return text1_final
