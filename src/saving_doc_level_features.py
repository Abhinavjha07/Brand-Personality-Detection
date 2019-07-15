#model training, saving the doc level features
!pip install imblearn
import os 
import numpy as np
import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
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
from imblearn.over_sampling import SMOTE
# /content/drive/My Drive/ML_Datasets/genData/text.pickle

#loading the pickled files
pickle_in = open("text_RUG.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("train_data_rug.pickle","rb")
data = pickle.load(pickle_in)

embedding_index = {}
#creating embedding matrix
with open('glove.6B.300d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs,'f',sep=' ')
        embedding_index[word] = coefs


#hyperparameters
vocab_size = 30000
batch_size = 128
embedding_dim = 300
max_len = 3000

#tokenized the texts, to form the numerical vector
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
train_sentences_tokenized = tokenizer.texts_to_sequences(X)

X = pad_sequences(train_sentences_tokenized, maxlen=max_len)
# print(X.shape)
# print(X)
# print(tokenizer.word_index)

#concatenating the binary labels of all personality traits
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
Y_rug = Y_4
Y_4 = to_categorical(Y_4)
Y_5 = label_enc.transform(data['cSOP_tag'])

Y_5 = to_categorical(Y_5)
Y = np.concatenate((Y_1,Y_2,Y_3,Y_4,Y_5),axis=1)
print(Y.shape)
# # for rugggedness only
sm = SMOTE(random_state=4991, n_jobs=8, ratio={1:2500, 0:2500})
new_X,new_Y = sm.fit_sample(X,Y_rug)
print(new_X.shape,new_Y.shape)
new_Y = to_categorical(np.reshape(new_Y,(-1,1)))
# ###############


#data splitting
train_X,test_X,train_Y,test_Y = train_test_split(new_X,new_Y,test_size = 0.1,random_state = 4991)

print('Preparing embedding matrix')
num_words = min(vocab_size,len(word_index))+1
embedding_matrix = np.zeros((num_words,embedding_dim))

for word,i in word_index.items():
    if i > vocab_size:
        continue

    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(num_words,
                            embedding_dim,
                            embeddings_initializer = Constant(embedding_matrix),
                            input_length = max_len,
                            trainable = False)

#creating the model
sequence_input = Input(shape = (max_len,),dtype = 'int32')
embedded_sequences = embedding_layer(sequence_input)
x1 = Conv1D(64,1,activation = 'relu')(embedded_sequences)
# x1 = GlobalMaxPooling1D()(x1)
x2 = Conv1D(64,2,activation = 'relu')(embedded_sequences)
# x2 = GlobalMaxPooling1D()(x2)
x3 = Conv1D(64,3,activation = 'relu')(embedded_sequences)
# x3 = GlobalMaxPooling1D()(x3)
print(x1.shape,x2.shape,x3.shape)
x = concatenate([x1,x2,x3],axis=1)
print(x.shape)
x = Dropout(0.5)(x)
x = Conv1D(128,3,activation = 'relu')(x)
docvec = GlobalMaxPooling1D()(x)
doc_model = Model(sequence_input,docvec)
x = Dropout(0.4)(docvec)
x = Dense(64,activation = 'relu')(x)
x = Dropout(0.5)(x)
pred = Dense(2,activation = 'softmax')(x)
model = Model(sequence_input,pred)
# model.summary()

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0001)

model.compile(loss = 'binary_crossentropy',optimizer = adam ,metrics = ['accuracy'])
model.summary()
a,b = 0,2
model.fit(train_X,train_Y[:,a:b],batch_size = batch_size,epochs = 100,validation_data = (test_X,test_Y[:,a:b]))
scores = model.evaluate(test_X, test_Y[:,a:b], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
plot_model(model, to_file='model.png',show_shapes = True)

model.save('my_model_cRUG_tag')

#saving the doc model(document level features)
doc_model.save('doc_model_cRUG')

pred = model.predict(test_X)
pred = pred.argmax(axis=1)
    
c_matrix = confusion_matrix(test_Y[:,a:b].argmax(axis=1),pred)
print(c_matrix)
accuracy = accuracy_score(test_Y[:,a:b].argmax(axis=1),pred)
print('Accuracy : ',accuracy)
#precision = true positive / total predicted positive(True positive + False positive)
#recall = true positive / total actual positive(True positive + False Negative)
print(classification_report(test_Y[:,a:b].argmax(axis=1),pred))

pickle_out = open("tokenizer.pickle","wb")
pickle.dump(tokenizer, pickle_out)
pickle_out.close()

