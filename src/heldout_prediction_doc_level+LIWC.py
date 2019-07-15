# heldout test data prediction(doc_level + LIWC features)

import os 
import numpy as np
import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,load_model
from keras.utils import to_categorical,plot_model
from keras.layers import Activation, Dense, Dropout,Input,Add,concatenate
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from keras.layers import Conv1D,MaxPooling1D,Embedding,GlobalMaxPooling1D
from keras.initializers import Constant
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score
import pickle

#loa
pickle_in = open("heldout_mrs_doc_cSOP.pickle","rb")
X = pickle.load(pickle_in)
print(X.shape)

pickle_in = open("/content/drive/My Drive/ML_Datasets/genData/test_data.pickle","rb")
data = pickle.load(pickle_in)
model = load_model('mrs_model_cSOP')

vocab_size = 30000
batch_size = 128
embedding_dim = 300
max_len = 3000

tags = ['y','n']
label_enc = LabelBinarizer()
label_enc.fit(tags)
Y_1 = label_enc.transform(data['cSIN'])

Y_2 = label_enc.transform(data['cEXC'])

Y_3 = label_enc.transform(data['cCOM'])

Y_4 = label_enc.transform(data['cRUG'])

Y_5 = label_enc.transform(data['cSOP'])


print('\n\nSophistication')

pred = model.predict(X)
pred = pred.argmax(axis=1)
Y = Y_5
c_matrix = confusion_matrix(Y,pred)
print(c_matrix)
accuracy = accuracy_score(Y,pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y,pred))
print(accuracy)
print(precision_score(Y,pred))
print(recall_score(Y,pred))
print(f1_score(Y,pred))

print('\n\nSincerity')

pickle_in = open("heldout_mrs_doc_cSIN.pickle","rb")
X = pickle.load(pickle_in)
print(X.shape)
model = load_model('mrs_model_cSIN')

pred = model.predict(X)
pred = pred.argmax(axis=1)
Y = Y_1
c_matrix = confusion_matrix(Y,pred)
print(c_matrix)
accuracy = accuracy_score(Y,pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y,pred))
print(accuracy)
print(precision_score(Y,pred))
print(recall_score(Y,pred))
print(f1_score(Y,pred))

print('\n\nExcitement')

model = load_model('mrs_model_cEXC')

pickle_in = open("heldout_mrs_doc_cEXC.pickle","rb")
X = pickle.load(pickle_in)
print(X.shape)

pred = model.predict(X)
pred = pred.argmax(axis=1)
Y = Y_2
c_matrix = confusion_matrix(Y,pred)
print(c_matrix)
accuracy = accuracy_score(Y,pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y,pred))
print(accuracy)
print(precision_score(Y,pred))
print(recall_score(Y,pred))
print(f1_score(Y,pred))


print('\n\nCompetence')
model = load_model('mrs_model_cCOM')

pickle_in = open("heldout_mrs_doc_cCOM.pickle","rb")
X = pickle.load(pickle_in)
print(X.shape)

pred = model.predict(X)
pred = pred.argmax(axis=1)
Y = Y_3
c_matrix = confusion_matrix(Y,pred)
print(c_matrix)
accuracy = accuracy_score(Y,pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y,pred))
print(accuracy)
print(precision_score(Y,pred))
print(recall_score(Y,pred))
print(f1_score(Y,pred))

print('\n\nRuggedness')
model = load_model('mrs_model_cRUG')

pickle_in = open("heldout_mrs_doc_cRUG.pickle","rb")
X = pickle.load(pickle_in)
print(X.shape)

pred = model.predict(X)
pred = pred.argmax(axis=1)
Y = Y_4
c_matrix = confusion_matrix(Y,pred)
print(c_matrix)
accuracy = accuracy_score(Y,pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y,pred))
print(accuracy)
print(precision_score(Y,pred))
print(recall_score(Y,pred))
print(f1_score(Y,pred))




