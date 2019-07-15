#concatenating the doc level features of dl-final model to LIWC features
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

file_names = file_names = ['com_mt_mrs_16thMay.csv','rug_mt_mrs_27thMay.csv','exc_mt_mrs_16thMay.csv','sinc_mt_mrs_16thMay.csv','sop_mt_mrs_16thMay.csv']
#loading the doc model
model = load_model('doc_model_cRUG')

#loading the pickled files
pickle_in = open("text_RUG.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("train_data_rug.pickle","rb")
data = pickle.load(pickle_in)



pickle_in = open("/content/drive/My Drive/ML_Datasets/mrs_column.pickle","rb")
col_names = pickle.load(pickle_in)
col_names = col_names[1:len(col_names)-1]

vocab_size = 30000
batch_size = 128
embedding_dim = 300
max_len = 3000

tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
train_sentences_tokenized = tokenizer.texts_to_sequences(X)
X = pad_sequences(train_sentences_tokenized, maxlen=max_len)
doc_vec = model.predict(X)
doc_vec = np.array(doc_vec)
print(doc_vec.shape)

#loading the mairesse features
mrs_features = pd.read_csv('/content/drive/My Drive/ML_Datasets/genData/MTdata/rug_mt_mrs_27thMay.csv',usecols = col_names)
print(mrs_features.shape)
# print(mrs_features.head())
mrs_features = np.array(mrs_features,dtype = 'float64')
print(mrs_features.shape)
#concatenating the LIWC features to doc-level features
doc_vec = np.concatenate((doc_vec,mrs_features),axis = 1)
print(doc_vec.shape)
pickle_out = open("mrs_doc_cRUG.pickle","wb")
pickle.dump(doc_vec, pickle_out)
pickle_out.close()

# col_names = []
# for col in mrs_features.columns:
#     col_names.append(col)

# print(col_names)
# pickle_out = open("mrs_column.pickle","wb")
# pickle.dump(col_names, pickle_out)
# pickle_out.close()

