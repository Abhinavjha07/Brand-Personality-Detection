#annotation sentences prediction

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


vocab_size = 30000
batch_size = 128
embedding_dim = 300
max_len = 3000
# content/drive/My Drive/ML_Datasets/genData/test_data.pickle
pickle_in = open("annote_sent.pickle","rb")
text = pickle.load(pickle_in)

pickle_in = open("annote_data.pickle","rb")
data = pickle.load(pickle_in)

pickle_in = open("tokenizer.pickle","rb")
tokenizer = pickle.load(pickle_in)
# tokenizer = Tokenizer(num_words = vocab_size)
# tokenizer.fit_on_texts(X)
# word_index = tokenizer.word_index
train_sentences_tokenized = tokenizer.texts_to_sequences(text)

X = pad_sequences(train_sentences_tokenized, maxlen=max_len)

tags = ['y','n']

label_enc = LabelBinarizer()
label_enc.fit(tags)
Y_1 = label_enc.transform(data['cSIN_label_fin'])
Y_1 = to_categorical(Y_1)

Y_2 = label_enc.transform(data['cEXC_label_fin'])
Y_2 = to_categorical(Y_2)

Y_3 = label_enc.transform(data['cCOM_label_fin'])
Y_3 = to_categorical(Y_3)

Y_4 = label_enc.transform(data['cRUG_label_fin'])
Y_4 = to_categorical(Y_4)

Y_5 = label_enc.transform(data['cSOP_label_fin'])
Y_5 = to_categorical(Y_5)

Y = np.concatenate((Y_1,Y_2,Y_3,Y_4,Y_5),axis=1)
print(Y.shape)

model1 = load_model('DL_Final_SIN2')
model2 = load_model('DL_Final_EXC2')
model3 = load_model('DL_Final_COM2')
model4 = load_model('DL_Final_RUG')
model5 = load_model('DL_Final_SOP2')


Y1 = model1.predict(X)
Y2 = model2.predict(X)
Y3 = model3.predict(X)
Y4 = model4.predict(X)
Y5 = model5.predict(X)
Y1 = Y1.argmax(axis=1) 
Y2 = Y2.argmax(axis=1) 
Y3 = Y3.argmax(axis=1) 
Y4 = Y4.argmax(axis=1)
Y5 = Y5.argmax(axis=1) 
# np.save('SIN_pred',Y1.argmax(axis=1))
# np.save('EXC_pred',Y2.argmax(axis=1))
# np.save('COM_pred',Y3.argmax(axis=1))
# np.save('RUG_pred',Y4.argmax(axis=1))
# np.save('SOP_pred',Y5.argmax(axis=1))


with open('output.csv','w') as f:
    f.write('Text , SIN , EXC , COM , RUG , SOP \n')
for i in range(len(text)):
    with open('output.csv','a') as f:
        f.write(text[i]+' , '+str(Y1[i])+' , '+str(Y2[i])+' , '+str(Y3[i])+' , '+str(Y4[i])+' , '+str(Y5[i])+' \n')


from itertools import islice
path = '/content/drive/My Drive/ML_Datasets/output.csv'
reader = csv.DictReader(open(path,encoding='latin-1'))
datalist = []
col_names = ['Text','SIN','EXC','COM','RUG','SOP']
for raw in reader:
    datalist.append((raw['Text '],raw[' SIN '],raw[' EXC '],raw[' COM '],raw[' RUG '],raw[' SOP ']))
data = pd.DataFrame.from_records(datalist, columns=col_names)  
# print(data)
with open('output_final.csv','w') as f:
    f.write('AUTH_ID,Text,SIN,EXC,COM,RUG,SOP \n')
c = 0
x = 0
for doc,sentences in auth_id.items():
    c += len(sentences)
    z = 0
    print(doc,x)
    for (sent,row) in zip(sentences,islice(data.itertuples(index=True, name='Pandas'),x,None)):
        
        x+=1
        print(str(x),sent,getattr(row, "Text"))
        
        with open('output_final.csv','a') as f:
            f.write(str(doc)+','+getattr(row, "Text")+','+getattr(row, "SIN")+','+getattr(row, "EXC")+','+getattr(row, "COM")+','+getattr(row, "RUG")+','+getattr(row, "SOP")+'\n')
            





