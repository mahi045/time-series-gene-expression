import numpy as np
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import KFold
from MachineLearning import  measure_error
import matplotlib.pyplot as plt
from Util import create_dataset,binary_convertor_annomaly_label
from sklearn.metrics import f1_score


DIM = 3
ACTUAL_LABEL = list(np.loadtxt('basic_label.txt'))
N_TIME = len(ACTUAL_LABEL[0])
N_GENE = len(ACTUAL_LABEL)
TOTAL_SAMPLE = N_TIME - DIM + 1
ACTUAL_LABEL = [[str(i) for i in l] for l in ACTUAL_LABEL]  # convert value to string as a label
TEST_PERCENT = 0.3
LAST_IDX = int(np.ceil(TOTAL_SAMPLE*(1-TEST_PERCENT)))


def convert_in_phase_space(series):
    #print(len(series))

    vector = []
    for i in range(DIM-1,len(series)):
        dim = []
        for j in range(DIM):
            dim.append(series[i-j])
        #print(dim)
        dim.reverse()
        #dim = projected_phase_space(vector=dim)  # specialized projection, eita diye o test korbo
        vector.append(dim)
    #print(len(vector))
    v = np.array(vector)
    return v


def single_sample_mlp(vector,index):
    clf = MLPClassifier(activation='tanh',solver='lbfgs',hidden_layer_sizes=(10,5,8),
                    learning_rate_init=0.1,max_iter=1000)
    X = vector
    Y = ACTUAL_LABEL[index][DIM-1:]
    X_train,X_test = X[:LAST_IDX], X[LAST_IDX:]
    Y_train,Y_test = Y[:LAST_IDX], Y[LAST_IDX:]

    clf.fit(X_train,Y_train)
    #print(clf.score(X_train,Y_train))
    y_predict = clf.predict(X_test)
    print(y_predict)
    print(Y_test)
    accuracy = accuracy_measure(y_predict,Y_test)
    print(accuracy)
    f1 = f1_score(Y_test,y_predict,average='weighted')
    print(f1)
    return accuracy


def accuracy_measure(predict,actual):
    x = np.zeros(len(predict))
    for i in range(len(predict)):
        x[i] = abs(float(predict[i]) - float(actual[i]))
    #print(x)
    acc = 1 - np.average(x)
    return acc


def combined_mlp(data):
    clf = MLPClassifier(activation='logistic', solver='lbfgs', hidden_layer_sizes=(10, 5, 8),
                        learning_rate_init=0.1, max_iter=100)
    X = np.array(data)
    Y = np.loadtxt('basic_label.txt')
    Y = binary_convertor_annomaly_label(Y)
    kf = KFold(n_splits=3)
    for train_idx,test_idx in kf.split(X):
        print(train_idx,test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        clf.fit(X_train, Y_train)
        #print(clf.score(X_train, Y_train))

        y_predict = clf.predict(X_test)
        for i in range(len(Y_test)):
            #print(y_predict[i])
            #print(Y_test[i])
            print(accuracy_measure(y_predict[i],Y_test[i]))
            print(f1_score(Y_test[i],y_predict[i]))
        #print(clf.score(X_test, Y_test))

