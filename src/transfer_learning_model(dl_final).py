#transfer learning
import os 
import numpy as np
import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,load_model,Sequential
from keras.utils import to_categorical
from keras.layers import Activation, Dense, Dropout,Input,Add,concatenate
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from keras.layers import Conv1D,MaxPooling1D,Embedding,GlobalMaxPooling1D
from keras.initializers import Constant
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score
import pickle
from imblearn.over_sampling import SMOTE


vocab_size = 30000
batch_size = 25
embedding_dim = 300
max_len = 3000

#loading the ht_test pickle
pickle_in = open("ht_test_text.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("ht_test_data.pickle","rb")
data = pickle.load(pickle_in)


# pickle_in = open("tokenizer.pickle","rb")
# tokenizer = pickle.load(pickle_in)
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
train_sentences_tokenized = tokenizer.texts_to_sequences(X)

X = pad_sequences(train_sentences_tokenized, maxlen=max_len)

tags = ['y','n']

#concatenating the binary labels to for n_data_samples*10 (2 for each label [y or n])
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

#7-fold validation, 6/7 for transfer learning and 1/7 for testing 
kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=4991)
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

#loading the models
model1 = load_model('my_model_cSIN_tag')

model2 = load_model('my_model_cEXC_tag')
model3 = load_model('my_model_cCOM_tag')
model4 = load_model('my_model_cSOP_tag')
model5 = load_model('my_model_cRUG_tag')
a,b = 0,2

#transfer learning on all the personality traits
print('\n\nSincerity')
    
for train, test in kfold.split(X, Y[:,a:b].argmax(axis=1)):
    for layer in model1.layers[:-3]:
        layer.trainable = False
    model = Sequential()
    model.add(model1)
    model.summary()
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) 
    model.fit(X[train],Y[train,a:b],epochs = 100,batch_size = batch_size,validation_data = (X[test],Y[test,a:b]))
    model.save('DL_Final_SIN2')
    pred = model.predict(X[test])
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
      
      for layer in model2.layers[:-3]:
          layer.trainable = False
      model = Sequential()
      model.add(model2)
      model.summary()
      adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0001)
      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
      model.fit(X[train],Y[train,a:b],epochs = 100,batch_size = batch_size,validation_data = (X[test],Y[test,a:b]))
      model.save('DL_Final_EXC2')
      pred = model.predict(X[test])
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
    
    for layer in model3.layers[:-3]:
        layer.trainable = False
    model = Sequential()
    model.add(model3)
    model.summary()
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) 
    model.fit(X[train],Y[train,a:b],epochs = 100,batch_size = batch_size,validation_data = (X[test],Y[test,a:b]))
    model.save('DL_Final_COM2')
    pred = model.predict(X[test])
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


sm = SMOTE(random_state=4991, n_jobs=8, ratio={1:500, 0:500})
for train, test in kfold.split(X, Y[:,a:b].argmax(axis=1)):
    new_Y = Y[train,a:b].argmax(axis=1)
    
    new_X,new_Y = sm.fit_sample(X[train],new_Y)
    new_Y = to_categorical(new_Y)

    for layer in model5.layers[:-3]:
        layer.trainable = False
    model = Sequential()
    model.add(model5)
    model.summary()
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) 
    model.fit(new_X,new_Y,epochs = 100,batch_size = batch_size,validation_data = (X[test],Y[test,a:b]))
    model.save('DL_Final_RUG2')
    pred = model.predict(X[test])
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
  
    for layer in model4.layers[:-3]:
        layer.trainable = False
    model = Sequential()
    model.add(model4)
    model.summary()
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) 
    model.fit(X[train],Y[train,a:b],epochs = 100,batch_size = batch_size,validation_data = (X[test],Y[test,a:b]))
    model.save('DL_Final_SOP2')
    pred = model.predict(X[test])
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
    
#printing the results of transfer learning
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





