import numpy as np
from sklearn import svm
from Util import create_dataset,binary_convertor_annomaly_label
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score,f1_score


DIM = 10
ACTUAL_LABEL = np.loadtxt('yeast_label.txt')
ACTUAL_LABEL = binary_convertor_annomaly_label(ACTUAL_LABEL)
N_TIME = len(ACTUAL_LABEL[0])
N_GENE = len(ACTUAL_LABEL)


# retrun a np array of n*DIM size
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
    # if DIM == 3:
    #     fig = plt.figure()
    #     fig.suptitle('3D view')
    #     ax = Axes3D(fig)
    #     ax.scatter(v[:,0],v[:,1],v[:,2])
    # if DIM == 2:
    #     plt.figure().suptitle('2D view')
    #     plt.scatter(v[:,0],v[:,1])
    #     plt.draw()
    return v


# project vector into diagonal v = (I - A*A_T)v
def projected_phase_space(vector):
    A = np.ones((DIM,DIM))
    I = np.identity(DIM)
    X = I - A/DIM
    Y = np.matmul(X,vector)
    return Y


def one_class_svm(vector):
    #print('ONE CLASS SVM')
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(vector)
    y_predict = clf.predict(vector)
    y_predict = np.concatenate((np.ones((DIM - 1)), y_predict), axis=0)
    anomolous_points = np.where(y_predict == -1)
    #print(anomolous_points)
    return anomolous_points


def isolation_forest(vector):
    clf = IsolationForest(max_samples=N_TIME-DIM+1)
    clf.fit(vector)
    y_predict = clf.predict(vector)
    y_predict = np.concatenate((np.ones((DIM - 1)), y_predict), axis=0)
    anomolous_points = np.where(y_predict == -1)
    #print(anomolous_points)
    return anomolous_points


def local_outlier_factor(vector):
    clf = LocalOutlierFactor()
    y_predict = clf.fit_predict(vector)
    y_predict = np.concatenate((np.ones((DIM - 1)), y_predict), axis=0)
    anomolous_points = np.where(y_predict == -1)
    #print(anomolous_points)
    return anomolous_points


def eliptic_envelop(vector):
    clf = EllipticEnvelope()
    clf.fit(vector)
    y_predict = clf.predict(vector)
    anomolous_points = np.where(y_predict == -1)
    #print(anomolous_points)
    return anomolous_points


def draw_plot(series,name,anomolous_points):
    plt.figure().suptitle(name)
    plt.plot(series, '-D', markevery=list(anomolous_points))


def measure_error(index,anomolous_points):
    x = np.zeros(N_TIME)
    anomolous_points = anomolous_points[0]
    #print(anomolous_points)
    for j in range(N_TIME):
        if j in anomolous_points:
            x[j] = 1 - ACTUAL_LABEL[index][j]
        else:
            x[j] = ACTUAL_LABEL[index][j]
    #print(np.average(x))
    return np.average(x)


def unsupervised_testing(vector_list):
    ACC = []
    F =[]
    # one class SVM
    error = np.zeros(N_GENE)
    accuracy = np.zeros(N_GENE)  # 2 class accuracy score list
    fscore = np.zeros(N_GENE)
    for i in range(N_GENE):
        anomolous_points = list(one_class_svm(vector_list[i]))
        predicted_values = np.zeros(N_TIME)
        np.put(a=predicted_values, ind=anomolous_points[0], v=1)  # put 1 for anomolous points, 0 for normal
        # print(predicted_values)
        # print(ACTUAL_LABEL[i])
        accuracy[i] = accuracy_score(y_true=ACTUAL_LABEL[i],y_pred=predicted_values)
        fscore[i] = f1_score(ACTUAL_LABEL[i],predicted_values)
        # print(fscore[i])
        #error[i] = measure_error(i, anomolous_points)
    #print('SVM total accuracy', 1 - np.average(error))
    print('SVM total accuracy', np.average(accuracy))
    print('SVM total fscore', np.average(fscore))
    ACC.append(np.average(accuracy))
    F.append(np.average(fscore))
    print('-------------------------------------------')
    # Isolation forest


    for i in range(N_GENE):
        anomolous_points = list(isolation_forest(vector_list[i]))
        predicted_values = np.zeros(N_TIME)
        np.put(a=predicted_values, ind=anomolous_points[0], v=1)  # put 1 for anomolous points, 0 for normal
        accuracy[i] = accuracy_score(y_true=ACTUAL_LABEL[i], y_pred=predicted_values)
        fscore[i] = f1_score(ACTUAL_LABEL[i], predicted_values)
        # print(fscore[i])
        #error[i] = measure_error(i, anomolous_points)
    #print('Isolation Forest total accuracy', 1 - np.average(error))
    print('Isolation Forest total accuracy', np.average(accuracy))
    print('Isolation Forest total fscore', np.average(fscore))
    ACC.append(np.average(accuracy))
    F.append(np.average(fscore))
    print('-------------------------------------------')
    # Local Outlier Factor

    for i in range(N_GENE):
        anomolous_points = list(local_outlier_factor(vector_list[i]))
        predicted_values = np.zeros(N_TIME)
        np.put(a=predicted_values, ind=anomolous_points[0], v=1)  # put 1 for anomolous points, 0 for normal
        accuracy[i] = accuracy_score(y_true=ACTUAL_LABEL[i], y_pred=predicted_values)
        fscore[i] = f1_score(ACTUAL_LABEL[i], predicted_values)
        # print(fscore[i])
        #error[i] = measure_error(i, anomolous_points)
    #print('Local Outlier total accuracy', 1 - np.average(error))
    print('Local Outlier total accuracy', np.average(accuracy))
    print('Local Outlier total fscore', np.average(fscore))
    ACC.append(np.average(accuracy))
    F.append(np.average(fscore))
    print('-------------------------------------------')
    # Elliptic envelop

    for i in range(N_GENE):
        anomolous_points = list(eliptic_envelop(vector_list[i]))
        predicted_values = np.zeros(N_TIME)
        np.put(a=predicted_values, ind=anomolous_points[0], v=1)  # put 1 for anomolous points, 0 for normal
        accuracy[i] = accuracy_score(y_true=ACTUAL_LABEL[i], y_pred=predicted_values)
        fscore[i] = f1_score(ACTUAL_LABEL[i], predicted_values)
        # print(fscore[i])
        #error[i] = measure_error(i, anomolous_points)
    #print('Elliptic Envelop total accuracy', 1 - np.average(error))
    print('Elliptic Envelop total accuracy', np.average(accuracy))
    print('Local Outlier total fscore', np.average(fscore))
    ACC.append(np.average(accuracy))
    F.append(np.average(fscore))
    print(ACC)
    print(F)
    print('-------------------------------------------')
    f = open('out.txt','a')
    f.write('2 class ACC F1 normal\n')
    for i in range(4):
        f.write(str(DIM)+' '+str(ACC[i])+' '+str(F[i]))
    f.write('\n')


def main():
    data = create_dataset()
    global  DIM
    for DIM in [2,3,4,5,6]:
        print('DIM: ',DIM)
        vector_list = []
        for i in range(N_GENE):
            vector = convert_in_phase_space(data[i])
            vector_list.append(vector)
        unsupervised_testing(vector_list)


        # draw_plot(series=data[i],name='ONE CLASS SVM', anomolous_points=anomolous_points)
    #   plt.show()
    return


def test_function():
    projected_phase_space(vector=None)
    pass

if __name__ == '__main__':
    main()
    #test_function()
    pass
