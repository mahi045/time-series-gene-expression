import numpy as np
from Util import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,SpatialDropout1D
from keras.layers import Conv1D, GlobalMaxPooling1D

gene_names, gene_types, data = get_patient_data()
X = data
Y = gene_types
X_train, X_test, labels_train,  labels_test = train_test_split(X,Y, test_size=0.2, random_state=None)


# get validation set
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train,
                                                test_size=0.1, random_state = None)
y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)

max_feature = 1   #
input_len = 9 # 9 time points
embed_dim = 1
batch_size= 32
num_of_class = 2   # 0 hoile poor, 1 hoile good

model = Sequential()
model.add(Embedding(max_feature, embed_dim, input_length=input_len, dropout=0.1,
                    ))
model.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
# model.add(Conv1D(filters=50, kernel_size=3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(num_of_class, activation='softmax'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

#Here we train the Network.

model.fit(X_tr, y_tr, batch_size =batch_size, nb_epoch = 20,  verbose = 2,validation_data=(X_vld,y_vld))

# Measuring score and accuracy on validation set

score,acc = model.evaluate(X_vld, y_vld, verbose = 0, batch_size = batch_size)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))

y_predict = model.predict(X_test)
for i in range(len(y_predict)):
    print(y_test[i],y_predict[i])

