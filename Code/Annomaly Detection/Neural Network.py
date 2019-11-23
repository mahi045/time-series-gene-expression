import numpy as np
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import KFold
from MachineLearning import  measure_error
import matplotlib.pyplot as plt
from Util import create_dataset,binary_convertor_annomaly_label
from sklearn.metrics import f1_score,accuracy_score

DIM = 3
K = 2
ACTUAL_LABEL = list(np.loadtxt('yeast_label.txt'))
N_TIME = len(ACTUAL_LABEL[0])
N_GENE = len(ACTUAL_LABEL)
TOTAL_SAMPLE = N_TIME - DIM + 1
ACTUAL_LABEL = [[str(i) for i in l] for l in ACTUAL_LABEL]  # convert value to string as a label
TEST_PERCENT = 0.4
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
    clf = MLPClassifier(activation='logistic',solver='lbfgs',hidden_layer_sizes=(10,5,8),
                    learning_rate_init=0.1,max_iter=1000)
    X = vector
    Y = ACTUAL_LABEL[index][DIM-1:]
    X_train,X_test = X[:LAST_IDX], X[LAST_IDX:]
    Y_train,Y_test = Y[:LAST_IDX], Y[LAST_IDX:]

    clf.fit(X_train,Y_train)
    #print(clf.score(X_train,Y_train))
    y_predict = clf.predict(X_test)
    # print('predicted',y_predict)
    # print('actual',Y_test)
    accuracy = accuracy_measure(y_predict,Y_test)
    print('Gene No:',index,' acc:',accuracy)
    f1 = f1_score(Y_test,y_predict,average='macro')
    print('f1:',f1)
    return accuracy,f1


def accuracy_measure(predict,actual):
    x = np.zeros(len(predict))
    for i in range(len(predict)):
        x[i] = abs(float(predict[i]) - float(actual[i]))
    #print(x)
    acc = 1 - np.average(x)
    return acc





def combined_mlp(data):
    clf = MLPClassifier(activation='logistic', solver='lbfgs', hidden_layer_sizes=(10, 5, 8),
                        learning_rate_init=0.1, max_iter=10000)
    X = np.array(data)
    Y = np.loadtxt('yeast_label.txt')
    Y = binary_convertor_annomaly_label(Y)
    kf = KFold(n_splits=K)
    train_acc,test_acc,train_f1,test_f1 = [],[],[],[]
    for train_idx,test_idx in kf.split(X):
        #print(train_idx,test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        clf.fit(X_train, Y_train)
        #print(clf.score(X_train, Y_train))

        y_predict_test = clf.predict(X_test)
        y_predict_train = clf.predict(X_train)

        acc = np.zeros(len(Y_train))
        f1 = np.zeros(len(Y_train))
        for i in range(len(Y_train)):
            acc[i]=accuracy_measure(y_predict_train[i],Y_train[i])
            f1[i]=f1_score(Y_train[i],y_predict_train[i])
        print('Train acc: ',np.average(acc))
        print('Train f1: ', np.average(f1))
        train_acc.append(np.average(acc))
        train_f1.append(np.average(f1))

        acc = np.zeros(len(Y_train))
        f1 = np.zeros(len(Y_train))
        for i in range(len(Y_test)):
            # print(y_predict_test[i])
            # print(Y_test[i])
            acc[i]=accuracy_measure(y_predict_test[i],Y_test[i])
            f1[i]=f1_score(Y_test[i],y_predict_test[i])
        print('Test acc: ', np.average(acc))
        print('Test f1: ', np.average(f1))
        test_acc.append(np.average(acc))
        test_f1.append(np.average(f1))
        print('=========================================')

    print('AVG Train acc: ', np.average(train_acc))
    print('AVG Test acc: ', np.average(test_acc))
    print('AVG Train f1: ', np.average(train_f1))

    print('AVG Test f1: ', np.average(test_f1))


def main():
    data = create_dataset()
    # #Single MLP Test
    # vector_list = []
    # for i in range(N_GENE):
    #     vector = convert_in_phase_space(data[i])
    #     vector_list.append(vector)
    #
    # print(LAST_IDX)
    # accuracy = np.zeros(N_GENE)
    # f1score = np.zeros(N_GENE)
    # for i in range(N_GENE):
    #     print('==========================================')
    #     accuracy[i],f1score[i] = single_sample_mlp(vector_list[i],index=i)
    # print('Single sample mlp accuracy ',np.average(accuracy))
    # print('Single sample mlp f1 score ', np.average(f1score))

    # # Combined MLP test
    combined_mlp(data)


if __name__ == '__main__':
    main()
