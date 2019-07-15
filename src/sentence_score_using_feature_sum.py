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

!pip3 install ortools
import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from numpy import linalg as LA
import math
from ortools.linear_solver import pywraplp


data = pd.read_csv('/content/drive/My Drive/ML_Datasets/toAnnotateDataPoints8thJune_final.csv')
print(data.shape)

site_content = data['site.content']
AUTH_ID = data['X.AUTHID']
docs = {}
for i in range(site_content.size):
    docs[i] = site_content[i]



original_corpus = [1]*len(docs)
corpus = [1]*len(docs)
copy_corpus = [1]*len(docs)

for i in range(len(docs)):
    original_corpus[i] = sent_tokenize(docs[i])
    corpus[i] = sent_tokenize(docs[i])
    copy_corpus[i] = sent_tokenize(docs[i])

def pos_score(pos_tag):
    return 1

def fam_score(fam):
    if(fam==0):
        return 1
    return 1/(1+np.exp( -8 * (-0.5 + 1/fam) ))

def position_score(pos):
    if(pos<=0.1):
        return 0.17
    elif(pos<=0.2):
        return 0.23
    elif(pos<=0.3):
        return 0.14
    elif(pos<=0.4):
        return 0.08
    elif(pos<=0.5):
        return 0.05
    elif(pos<=0.6):
        return 0.04
    elif(pos<=0.7):
        return 0.06
    elif(pos<=0.8):
        return 0.04
    elif(pos<=0.9):
        return 0.04
    else:
        return 0.15


count_word = [1]*len(corpus)
count_matrix = [1]*len(corpus)
count_persent = [1]*len(corpus) # no of words per sentence

vectorizer = CountVectorizer()
for i in range(len(corpus)):
    count_word[i] = vectorizer.fit_transform(corpus[i])
    count_matrix[i] = count_word[i].toarray()
    count_persent[i] = np.sum(count_matrix[i], axis=1)


from nltk.stem import LancasterStemmer, WordNetLemmatizer
import re, string, unicodedata
def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words
  

  
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
#     words = remove_stopwords(words)
#     words = lemmatize_verbs(words)
    return words
  
lemmatizer = WordNetLemmatizer()  
def lemm_tokens(tokens):
    
    return [lemmatizer.lemmatize(item) for item in tokens]
  
from collections import Counter
import csv
mu = 0.85
lam = 0.7
k = 3

top_corpus = [1]*len(corpus)
score_sentences = []
for d in range(len(corpus)):

    
    no_sent = len(corpus[d])
    document = ' '.join(corpus[d])
    
    
    words = nltk.tokenize.word_tokenize(document)
    words = remove_stopwords(words)
    words = to_lowercase(words)
    words = lemmatize_verbs(words)
    
    
    fdist1 = Counter(words)

    tf_word = {word:freq for word, freq in fdist1.items() if not word.isdigit()}
    
    # length of word
    len_word = {word: len(word) for word in fdist1.keys()}
    
    # pos tag
    pos_word = {word:pos_score(nltk.pos_tag([word])[0][1]) for word in fdist1.keys()}
    
    # familiarity
    fam_word = {word: fam_score(len(wn.synsets(word))) for word in fdist1.keys()}
    
    score_sents = []
    for i in range(no_sent):
        score_sent = 0
        
        
        
        for j,word in enumerate(nltk.tokenize.word_tokenize(corpus[d][i])):
            # occurence score
            word = word.lower()
            if(word[len(word)-1]=='.'):
                    word = word[:len(word)-1]
            if word not in fdist1:
                continue
            
            if(not word.isdigit() and len(word)>1 and word !="''"):
                
                occ_word = position_score(j/len(corpus[d][i]))
            
                score_word = tf_word[word] * len_word[word] * pos_word[word] * fam_word[word] * occ_word
                score_sent += score_word
            
            # corefferant
            if(word !="''" and len(word)>1 and pos_word[word]=='PRP'):
                if(j/len(corpus[s][i]) < 0.5):
                    if(i>0):
                        score_sent += score_sents[i-1]/len(nltk.tokenize.word_tokenize(corpus[d][i-1]))
                else:
                    score_sent += score_sent/len(nltk.tokenize.word_tokenize(corpus[d][i]))
        
        score_sents.append(score_sent)
    score_sentences.append(score_sents)

print(len(score_sentences))

with open("FEA_sent_scores.csv",'w') as f:
    f.write('AUTH_ID , Text , Scores\n')
    
z= 0
with open("FEA_sent_scores.csv",'a') as f:
    for i in range(len(corpus)):
        z += len(corpus[i])
        for j in range(len(corpus[i])):
            word = nltk.word_tokenize(corpus[i][j])
            word = normalize(word) 
            words = ' '.join(word)
            f.write(AUTH_ID[i]+' , '+str(words)+' , '+str(score_sentences[i][j])+'\n')
        
    
print(z)

top_corpus = [1]*len(corpus)

# cross method
sentence_score = []
for s in range(len(corpus)):
# if(True):
#     s = 2
    
    no_sent = len(corpus[s])
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus[s])
    word_matrix = np.array(X.toarray())
    
    U, S, Vh = np.linalg.svd(word_matrix.T, full_matrices=False)

    assert np.prod([[np.abs(Vh[i][j]) <= 1.001 for i in range(no_sent)] for j in range(no_sent)])==1, print(Vh, [[np.abs(Vh[i][j]) <= 1 for i in range(no_sent)] for j in range(no_sent)])
#     print(Vh)
    Vh = Vh*(Vh>=np.mean(Vh, axis=0))*(Vh>=0)
#     print(Vh)
    
    length_sent = S.T.dot(Vh)
    sentence_score.append(length_sent)
    idx_top_sent = np.argsort(length_sent)
    
    
    select_sents = np.zeros(no_sent)
    for i in range(k):
        select_sents[idx_top_sent[no_sent-i-1]] = 1
    
#     print(select_sents, idx_top_sent)
    top_sents = []
    for i in range(no_sent):
        j=0
        while(j<select_sents[i]):
            top_sents.append(copy_corpus[s][i])
            j+=1
            
    top_corpus[s] = top_sents
print(top_corpus)
print(len(sentence_score))


with open("LSA_sent_scores.csv",'w') as f:
    f.write('AUTH_ID , Text , Scores\n')
z= 0
with open("LSA_sent_scores.csv",'a') as f:
    for i in range(len(corpus)):
        z += len(corpus[i])
        for j in range(len(corpus[i])):
            word = nltk.word_tokenize(corpus[i][j])
            word = normalize(word) 
            words = ' '.join(word)
            f.write(AUTH_ID[i]+' , '+str(words)+' , '+str(sentence_score[i][j])+'\n')
        
    
print(z)
