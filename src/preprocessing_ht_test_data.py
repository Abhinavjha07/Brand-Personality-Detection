#Preprocessing the ht_test data

import os 
import numpy as np
import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical,plot_model
from keras.layers import Activation, Dense, Dropout,Input,Add,concatenate
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from keras.layers import Conv1D,MaxPooling1D,Embedding,GlobalMaxPooling1D
from keras.initializers import Constant
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
from nltk.corpus import stopwords
import os
import nltk
nltk.download('all')
nltk.download('wordnet')
from tqdm import tqdm
import numpy as np
import os
import sys
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
import re, string, unicodedata
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


#preprocessing methods
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    #words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    return words
# content/drive/My Drive/ML_Datasets/genData/ht4_essays_data_12thFeb.csv

#path of the hr_test data
test_path = '/content/drive/My Drive/ML_Datasets/genData/ht4_essays_data_12thFeb.csv'
col_names = ['X.AUTHID','TEXT','cSIN','cEXC','cCOM','cRUG','cSOP']

#reading the data
reader = csv.DictReader(open(test_path,encoding='latin-1'))

#creating the test data list
datalist = []
for raw in reader:
    datalist.append((raw['X.AUTHID'],raw['TEXT'],raw['cSIN'],raw['cEXC'],raw['cCOM'],raw['cRUG'],raw['cSOP']))

train_data = np.array(datalist)
data = pd.DataFrame.from_records(datalist, columns=col_names)
print(train_data.shape)

pickle_out = open("ht_test_data.pickle","wb")
pickle.dump(data, pickle_out)
pickle_out.close()


#preprocessing the texts
words = []
for texts in data['TEXT']:
    text = denoise_text(texts)
    text = replace_contractions(text)
    word = nltk.word_tokenize(text)
    word = normalize(word)
    words.append(word)

words = np.array(words)
print(words.shape)
new_text = []
for i in range(len(words)):
    text = " ".join(str(x) for x in words[i])
    new_text.append(text)

print(len(new_text))


#saving the test file
pickle_out = open("ht_test_text.pickle","wb")
pickle.dump(new_text, pickle_out)
pickle_out.close()






