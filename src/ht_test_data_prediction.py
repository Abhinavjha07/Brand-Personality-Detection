#ht_test_data prediction
import os 
import numpy as np
import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,load_model
from keras.utils import to_categorical
from keras.layers import Activation, Dense, Dropout,Input,Add,concatenate
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from keras.layers import Conv1D,MaxPooling1D,Embedding,GlobalMaxPooling1D
from keras.initializers import Constant
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score
import pickle

#hyperparameters
vocab_size = 30000
batch_size = 128
embedding_dim = 300
max_len = 3000

# content/drive/My Drive/ML_Datasets/genData/test_data.pickle

#loading the pickled test data file
pickle_in = open("ht_test_text.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("ht_test_data.pickle","rb")
data = pickle.load(pickle_in)


#loading the pickled tokenizer
pickle_in = open("tokenizer.pickle","rb")
tokenizer = pickle.load(pickle_in)
# tokenizer = Tokenizer(num_words = vocab_size)
# tokenizer.fit_on_texts(X)
# word_index = tokenizer.word_index

#tokenizing the text data
train_sentences_tokenized = tokenizer.texts_to_sequences(X)

#padding the sequences
X = pad_sequences(train_sentences_tokenized, maxlen=max_len)

tags = ['y','n']


label_enc = LabelBinarizer()
label_enc.fit(tags)

Y_1 = label_enc.transform(data['cSIN'])
Y_1 = to_categorical(Y_1)

Y_2 = label_enc.transform(data['cEXC'])
Y_2 = to_categorical(Y_2)

Y_3 = label_enc.transform(data['cCOM'])
Y_3 = to_categorical(Y_3)

Y_4 = label_enc.transform(data['cRUG'])
Y_4 = to_categorical(Y_4)

Y_5 = label_enc.transform(data['cSOP'])
Y_5 = to_categorical(Y_5)

#concatenating the binary laabels to for n_data_samples*10 (2 for each label [y or n])
Y = np.concatenate((Y_1,Y_2,Y_3,Y_4,Y_5),axis=1)
print(Y.shape)



# 'cSIN','cEXC','cCOM','cRUG','cSOP'

#performing he seven fold validation
kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=4991)

#list to store the accuracy, precision, recall score and f1-score for the 7-fold validation
acscores1 = []
acscores2 = []
acscores3 = []
acscores4 = []
acscores5 = []

prescores1 = []
prescores2 = []
prescores3 = []
prescores4 = []
prescores5 = []

rescores1 = []
rescores2 = []
rescores3 = []
rescores4 = []
rescores5 = []
fscores1 = []
fscores2 = []
fscores3 = []
fscores4 = []
fscores5 = []

#loading the trained models
model1 = load_model('my_model_cSIN_tag')
model2 = load_model('my_model_cEXC_tag')
model3 = load_model('my_model_cCOM_tag')
model4 = load_model('my_model_cSOP_tag')
model5 = load_model('my_model_cRUG_tag')


#performing the 7-fold validation for all the personality traits
a,b = 0,2
print('\n\nSincerity')
    
for train, test in kfold.split(X, Y[:,a:b].argmax(axis=1)):  
    pred = model1.predict(X[test])
    pred = pred.argmax(axis=1)
    c_matrix = confusion_matrix(Y[test,a:b].argmax(axis=1),pred)
    print(c_matrix)
    accuracy = accuracy_score(Y[test,a:b].argmax(axis=1),pred)
    print('Accuracy : ',accuracy)
    # precision = true positive / total predicted positive(True positive + False positive)
    # recall = true positive / total actual positive(True positive + False Negative)
    print(classification_report(Y[test,a:b].argmax(axis=1),pred))
    acscores1.append(accuracy)
    prescores1.append(precision_score(Y[test,a:b].argmax(axis=1),pred))
    rescores1.append(recall_score(Y[test,a:b].argmax(axis=1),pred))
    fscores1.append(f1_score(Y[test,a:b].argmax(axis=1),pred))

a,b = 2,4

print('\n\nExcitement')
for train, test in kfold.split(X, Y[:,a:b].argmax(axis=1)):   
      pred = model2.predict(X[test])
      pred = pred.argmax(axis=1)
      c_matrix = confusion_matrix(Y[test,a:b].argmax(axis=1),pred)
      print(c_matrix)
      accuracy = accuracy_score(Y[test,a:b].argmax(axis=1),pred)
      print('Accuracy : ',accuracy)
      # # precision = true positive / total predicted positive(True positive + False positive)
      # # recall = true positive / total actual positive(True positive + False Negative)
      print(classification_report(Y[test,a:b].argmax(axis=1),pred))
      acscores2.append(accuracy)
      prescores2.append(precision_score(Y[test,a:b].argmax(axis=1),pred))
      rescores2.append(recall_score(Y[test,a:b].argmax(axis=1),pred))
      fscores2.append(f1_score(Y[test,a:b].argmax(axis=1),pred))

a,b = 4,6
print('\n\nCompetence')
for train, test in kfold.split(X, Y[:,a:b].argmax(axis=1)):   
    
    pred = model3.predict(X[test])
    pred = pred.argmax(axis=1)
    c_matrix = confusion_matrix(Y[test,a:b].argmax(axis=1),pred)
    print(c_matrix)
    accuracy = accuracy_score(Y[test,a:b].argmax(axis=1),pred)
    print('Accuracy : ',accuracy)
    # precision = true positive / total predicted positive(True positive + False positive)
    # recall = true positive / total actual positive(True positive + False Negative)
    print(classification_report(Y[test,a:b].argmax(axis=1),pred))
    acscores3.append(accuracy)
    prescores3.append(precision_score(Y[test,a:b].argmax(axis=1),pred))
    rescores3.append(recall_score(Y[test,a:b].argmax(axis=1),pred))
    fscores3.append(f1_score(Y[test,a:b].argmax(axis=1),pred))
    
a,b = 6,8
print('\n\nRuggedness')
for train, test in kfold.split(X, Y[:,a:b].argmax(axis=1)):   
    
    pred = model5.predict(X[test])
    pred = pred.argmax(axis=1)
    c_matrix = confusion_matrix(Y[test,a:b].argmax(axis=1),pred)
    print(c_matrix)
    accuracy = accuracy_score(Y[test,a:b].argmax(axis=1),pred)
    print('Accuracy : ',accuracy)
    # precision = true positive / total predicted positive(True positive + False positive)
    # recall = true positive / total actual positive(True positive + False Negative)
    print(classification_report(Y[test,a:b].argmax(axis=1),pred))
    acscores5.append(accuracy)
    prescores5.append(precision_score(Y[test,a:b].argmax(axis=1),pred))
    rescores5.append(recall_score(Y[test,a:b].argmax(axis=1),pred))
    fscores5.append(f1_score(Y[test,a:b].argmax(axis=1),pred))

a,b = 8,10

print('\n\nSophistication')
    
for train, test in kfold.split(X, Y[:,a:b].argmax(axis=1)): 
    pred = model4.predict(X[test])
    pred = pred.argmax(axis=1)
    c_matrix = confusion_matrix(Y[test,a:b].argmax(axis=1),pred)
    print(c_matrix)
    accuracy = accuracy_score(Y[test,a:b].argmax(axis=1),pred)
    print('Accuracy : ',accuracy)
    # precision = true positive / total predicted positive(True positive + False positive)
    # recall = true positive / total actual positive(True positive + False Negative)
    print(classification_report(Y[test,a:b].argmax(axis=1),pred))
    acscores4.append(accuracy)
    prescores4.append(precision_score(Y[test,a:b].argmax(axis=1),pred))
    rescores4.append(recall_score(Y[test,a:b].argmax(axis=1),pred))
    fscores4.append(f1_score(Y[test,a:b].argmax(axis=1),pred))

#printing the output for all the personaliy traits
print('\n\nSincerity')   
print(acscores1,'\nMean : ',np.mean(acscores1),'\nStandard deviation : ',np.std(acscores1),'\nPrecision Score : ',np.mean(prescores1),'\nRecall Score : ',np.mean(rescores1),'\nF1 Score : ',np.mean(fscores1))
print('\n\nExcitement')
print(acscores2,'\nMean : ',np.mean(acscores2),'\nStandard deviation : ',np.std(acscores2),'\nPrecision Score : ',np.mean(prescores2),'\nRecall Score : ',np.mean(rescores2),'\nF1 Score : ',np.mean(fscores2))
print('\n\nCompetence')
print(acscores3,'\nMean : ',np.mean(acscores3),'\nStandard deviation : ',np.std(acscores3),'\nPrecision Score : ',np.mean(prescores3),'\nRecall Score : ',np.mean(rescores3),'\nF1 Score : ',np.mean(fscores3))
print('\n\nSophistication')
print(acscores4,'\nMean : ',np.mean(acscores4),'\nStandard deviation : ',np.std(acscores4),'\nPrecision Score : ',np.mean(prescores4),'\nRecall Score : ',np.mean(rescores4),'\nF1 Score : ',np.mean(fscores4))

print('\n\nRuggedness')
print(acscores5,'\nMean : ',np.mean(acscores5),'\nStandard deviation : ',np.std(acscores5),'\nPrecision Score : ',np.mean(prescores5),'\nRecall Score : ',np.mean(rescores5),'\nF1 Score : ',np.mean(fscores5))



