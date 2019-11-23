import numpy as np
from Util import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
import os
from f1_score_callback import f1_m
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from keras.layers import Conv1D, Conv2D, GlobalMaxPooling1D, MaxPooling1D, MaxPooling2D, Flatten

gene_names, gene_types, data = get_patient_data(file='../Datasets/GSE20305_series_matrix.txt')
X = data
Y = gene_types
X_train, X_test, labels_train,  labels_test = train_test_split(X,Y, test_size=0.2, random_state=None)

num_of_class = 5

# get validation set

X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train,
                                                test_size=0.1, random_state = None)
y_tr = label(lab_tr, num_of_class)
y_vld = label(lab_vld, num_of_class)
y_test = label(labels_test, num_of_class)

max_feature = 2   #
input_len = 8 # 9 time points
embed_dim = 1
batch_size= 32
  # 0 hoile poor, 1 hoile good
X_tr = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1], 1)
X_vld = X_vld.reshape(X_vld.shape[0], 1, X_vld.shape[1], 1)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], 1)

model = Sequential()
model.add(Conv2D(filters=30, kernel_size=(1, 3), padding='valid', activation='relu', strides=(1, 1)
          , input_shape=(X_tr.shape[1], X_tr.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(1, 2)))

# model.add(Conv2D(filters=40, kernel_size=(1, 4), padding='valid', activation='relu', strides=(1, 1)))
# model.add(MaxPooling2D(pool_size=(1, 2)))


# model.add(LSTM(units = 50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(GlobalMaxPooling1D())
# model.add(Conv1D(filters=80, kernel_size=2, padding='valid', activation='relu', strides=1))
# model.add(GlobalMaxPooling1D())
# model.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(num_of_class, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy', f1_m])
print(model.summary())

#Here we train the Network.
model.fit(X_tr, y_tr, batch_size =batch_size, epochs = 50,  verbose = 2,validation_data=(X_vld,y_vld))

# Measuring score and accuracy on validation set

score, acc, f1_score = model.evaluate(X_test, y_test, verbose = 0, batch_size = batch_size)
print("Logloss score: %.4f" % (score))
print("Validation set Accuracy: %.4f" % (acc))
print("F1 score: %.4f" % (f1_score))

y_predict = model.predict(X_test)
# for i in range(len(y_predict)):
#     print(y_test[i],y_predict[i])


