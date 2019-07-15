#splitting the texts into sentences
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import csv
import pandas as pd
import pickle

stemmer = nltk.stem.porter.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
remove_punctuation_map = dict((ord(char), ' ') for char in string.punctuation)
integer_chars = [str(digit) for digit in list(range(10))]
remove_integer_map = dict((ord(char), None) for char in integer_chars)

def sentsplit(docs):
    '''
    Splits each document of a document list(collection) into sentences and returns the split list
    '''
    return [nltk.sent_tokenize(doc) for doc in docs]

def stem_tokens(tokens):
    '''
    Transforms a list of words into a list of stemmed words
    '''
    return [stemmer.stem(item) for item in tokens]

def lemm_tokens(tokens):
    '''
    Transforms a list of words into a list of lemmatised words
    '''
    return [lemmatizer.lemmatize(item) for item in tokens]

def filtersent(text, remove_stopwords=True, stem=False, lemm=True, avoid_single_char=True):
    '''
    Filters a sentence according to requirements
    '''
    # remove puctuations and lower the case
    simpletext = text.lower().translate(remove_punctuation_map).translate(remove_integer_map)
    # tokenize
    words = nltk.word_tokenize(simpletext)
    # remove stop words
    if remove_stopwords:
        words = [w for w in words if w not in stopwords.words('english')]
    # lemmatize them
    if lemm:
        words = lemm_tokens(words)
    # stem them
    if stem:
        words = stem_tokens(words)
    # avoid single character words
    if avoid_single_char:
        words = [w for w in words if w not in string.ascii_lowercase]
    # detokenise the sentence
    return nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(words)

path = '/content/drive/My Drive/ML_Datasets/toAnnotateDataPoints7thJune.csv'

reader = csv.DictReader(open(path,encoding='latin-1'))
        
datalist = []
col_names = ['X.AUTHID','site.content','cSIN_label_fin','cEXC_label_fin','cCOM_label_fin','cRUG_label_fin','cSOP_label_fin']
for raw in reader:
    datalist.append((raw['X.AUTHID'],raw['site.content'],raw['cSIN_label_fin'],raw['cEXC_label_fin'],raw['cCOM_label_fin'],raw['cRUG_label_fin'],raw['cSOP_label_fin']))
data = pd.DataFrame.from_records(datalist, columns=col_names)  
text = data['site.content']

for t in text:
  print(t)
  
print('------------------------------------------------------------------------------------\n\n')
sentences = sentsplit(text)
for sent in sentences:
  print(sent)

X = []
auth_id = {}
for (doc,a) in zip(sentences,data['X.AUTHID']):
    s = []
    for sent in doc:
        x = filtersent(sent)
        X.append(x)
        s.append(x)
    
    auth_id[a] = s

# X = np.array(X)
# X = np.reshape(X,(-1,1))
# print(X.shape)
pickle_out = open("annote_data.pickle","wb")
pickle.dump(data, pickle_out)
pickle_out.close()
pickle_out = open("annote_sent.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()



