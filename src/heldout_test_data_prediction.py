#heldout_test_data prediction
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


#parameters same as the parameters which used for training the model
vocab_size = 30000
batch_size = 128
embedding_dim = 300
max_len = 3000

# /content/drive/My Drive/ML_Datasets/genData/test_data.pickle

#loading the pickled test data
pickle_in = open("test_text.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("test_data.pickle","rb")
data = pickle.load(pickle_in)


#loading the tokenizer pickled during the training of the model
pickle_in = open("tokenizer.pickle","rb")
tokenizer = pickle.load(pickle_in)
# tokenizer = Tokenizer(num_words = vocab_size)
# tokenizer.fit_on_texts(X)
# word_index = tokenizer.word_index

#tokenizing the texts
train_sentences_tokenized = tokenizer.texts_to_sequences(X)

#padding the sequence to have the pre decided length 
X = pad_sequences(train_sentences_tokenized, maxlen=max_len)

tags = ['y','n']

#concatenating the binary laabels to for n_data_samples*10 (2 for each binary label)

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

Y = np.concatenate((Y_1,Y_2,Y_3,Y_4,Y_5),axis=1)
print(Y.shape)



# 'cSIN','cEXC','cCOM','cRUG','cSOP'



#loading the trained models
model1 = load_model('my_model_cSIN_tag')
model2 = load_model('my_model_cEXC_tag')
model3 = load_model('my_model_cCOM_tag')
model4 = load_model('my_model_cSOP_tag')
model5 = load_model('my_model_cRUG_tag')


#making test data prediction for all the personality traits
a,b = 0,2
print('\n\nSincerity')

pred = model1.predict(X)
pred = pred.argmax(axis=1)
c_matrix = confusion_matrix(Y[:,a:b].argmax(axis=1),pred)
print(c_matrix)
accuracy = accuracy_score(Y[:,a:b].argmax(axis=1),pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y[:,a:b].argmax(axis=1),pred))


a,b = 2,4

print('\n\nExcitement')

    
pred = model2.predict(X)
pred = pred.argmax(axis=1)
c_matrix = confusion_matrix(Y[:,a:b].argmax(axis=1),pred)
print(c_matrix)
accuracy = accuracy_score(Y[:,a:b].argmax(axis=1),pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y[:,a:b].argmax(axis=1),pred))


a,b = 4,6
print('\n\nCompetence')
  
    
pred = model3.predict(X)
pred = pred.argmax(axis=1)
c_matrix = confusion_matrix(Y[:,a:b].argmax(axis=1),pred)
print(c_matrix)
accuracy = accuracy_score(Y[:,a:b].argmax(axis=1),pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y[:,a:b].argmax(axis=1),pred))


a,b = 6,8

print('\n\nRuggedness')

    
pred = model5.predict(X)
pred = pred.argmax(axis=1)
c_matrix = confusion_matrix(Y[:,a:b].argmax(axis=1),pred)
print(c_matrix)
accuracy = accuracy_score(Y[:,a:b].argmax(axis=1),pred)
print('Accuracy : ',accuracy)
print('Precision : ',precision_score(Y[:,a:b].argmax(axis=1),pred))
print('Recall : ',recall_score(Y[:,a:b].argmax(axis=1),pred))
print('F1 score : ',f1_score(Y[:,a:b].argmax(axis=1),pred))
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y[:,a:b].argmax(axis=1),pred))

a,b = 8,10

print('\n\nSophistication')

pred = model4.predict(X)
pred = pred.argmax(axis=1)
c_matrix = confusion_matrix(Y[:,a:b].argmax(axis=1),pred)
print(c_matrix)
accuracy = accuracy_score(Y[:,a:b].argmax(axis=1),pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(Y[:,a:b].argmax(axis=1),pred))
