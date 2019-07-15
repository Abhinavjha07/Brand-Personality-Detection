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
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
from imblearn.over_sampling import SMOTE

#loading the pickled concatenated doc level features
pickle_in = open("mrs_doc_cRUG.pickle","rb")
X = pickle.load(pickle_in)
print(X.shape)

#loading the train data
pickle_in = open("train_data_rug.pickle","rb")
data = pickle.load(pickle_in)


vocab_size = 30000
batch_size = 128
embedding_dim = 300
max_len = 3000

tags = ['y','n']
label_enc = LabelBinarizer()
label_enc.fit(tags)
Y_1 = label_enc.transform(data['cSIN_tag'])
Y_1 = to_categorical(Y_1)

Y_2 = label_enc.transform(data['cEXC_tag'])
Y_2 = to_categorical(Y_2)
Y_3 = label_enc.transform(data['cCOM_tag'])
Y_3 = to_categorical(Y_3)
Y_4 = label_enc.transform(data['cRUG_tag'])

Y_4 = to_categorical(Y_4)

Y_5 = label_enc.transform(data['cSOP_tag'])

Y_5 = to_categorical(Y_5)

#conactenating the binary labels
Y = np.concatenate((Y_1,Y_2,Y_3,Y_4,Y_5),axis=1)
print(Y.shape)

#train and validation split
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.1,random_state = 4991)


doc_shape = X.shape[1]

#creatng the model consisting of just the classification layers
inp = Input(shape = (doc_shape,),dtype = 'float32')
x = Dropout(0.4)(inp)
x = Dense(64,activation = 'relu')(x)
x = Dropout(0.5)(x)
pred = Dense(2,activation = 'softmax')(x)
model = Model(inp,pred)

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0001)

model.compile(loss = 'binary_crossentropy',optimizer = 'adam' ,metrics = ['accuracy'])
model.summary()
a,b = 6,8 #it decides on which trait we are training our model or testing
model.fit(train_X,train_Y[:,a:b],batch_size = batch_size,epochs = 100,validation_data = (test_X,test_Y[:,a:b]))
scores = model.evaluate(test_X, test_Y[:,a:b], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#saving the model
model.save('mrs_model_cRUG')
pred = model.predict(test_X)
pred = pred.argmax(axis=1)
    
#printing the classification metrics
c_matrix = confusion_matrix(test_Y[:,a:b].argmax(axis=1),pred)
print(c_matrix)
accuracy = accuracy_score(test_Y[:,a:b].argmax(axis=1),pred)
print('Accuracy : ',accuracy)
#precision = true positive / total predicted positive(True positive + False positive)
#recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(test_Y[:,a:b].argmax(axis=1),pred))




