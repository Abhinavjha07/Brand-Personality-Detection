#transfer learning on heldout

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

#loading the pickled test files
pickle_in = open("ht_test_text.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("ht_test_data.pickle","rb")
data = pickle.load(pickle_in)

test_X = pickle.load(open('/content/drive/My Drive/ML_Datasets/genData/test_text.pickle',"rb"))
test_data = pickle.load(open('/content/drive/My Drive/ML_Datasets/genData/test_data.pickle',"rb"))

#tokenizing the text
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
train_sentences_tokenized = tokenizer.texts_to_sequences(X)

X = pad_sequences(train_sentences_tokenized, maxlen=max_len)

test_sentences_tokenized = tokenizer.texts_to_sequences(test_X)

test_X = pad_sequences(test_sentences_tokenized, maxlen=max_len)

tags = ['y','n']

label_enc = LabelBinarizer()
label_enc.fit(tags)
Y = label_enc.transform(data['cRUG'])

#oversampling using SMOTE
sm = SMOTE(random_state=4991, n_jobs=8, ratio={1:500, 0:500})
X,Y = sm.fit_sample(X,Y)
Y = to_categorical(Y)

test_Y = label_enc.transform(test_data['cRUG'])

test_Y = to_categorical(test_Y)

#loading the model
model5 = load_model('my_model_cRUG_tag');
a,b = 0,2
print('\n\nRuggedness')


#setting the layers non-trainable except the last 3 
for layer in model5.layers[:-3]:
    layer.trainable = False


model = Sequential()
model.add(model5)
model.summary()

#saving the model
model.save('DL_Final_RUG')
adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) 
model.fit(X,Y,epochs = 100,batch_size = batch_size,validation_data = (test_X,test_Y))
pred = model.predict(test_X)
pred = pred.argmax(axis=1)
c_matrix = confusion_matrix(test_Y.argmax(axis=1),pred)
print(c_matrix)
accuracy = accuracy_score(test_Y.argmax(axis=1),pred)
print('Accuracy : ',accuracy)
# precision = true positive / total predicted positive(True positive + False positive)
# recall = true positive / total actual positive(True positive + False Negative)

#printing the classification metrics
print(classification_report(test_Y.argmax(axis=1),pred))
print(accuracy)
print(precision_score(test_Y.argmax(axis=1),pred))
print(recall_score(test_Y.argmax(axis=1),pred))
print(f1_score(test_Y.argmax(axis=1),pred))



