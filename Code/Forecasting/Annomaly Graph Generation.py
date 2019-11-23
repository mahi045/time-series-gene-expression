import matplotlib.pyplot as plt
import numpy as np

def supervised_single():
    A = np.loadtxt('data.txt')
    print(A)
    x = [10, 20, 30, 40]
    x = np.array(x)
    plt.bar(x,A,width=3)
    plt.ylim(0, 1)
    plt.xticks(x)
    plt.xlabel('Test Percent')
    plt.ylabel('Accuracy')
    #plt.legend(loc='best', ncol=2)
    plt.show()

def supervised_combined():
    A = np.loadtxt('data.txt')

    result = []
    x = [10,20,30,40,50]
    x = np.array(x)
    labels = ['Test','Train']
    for i in range(len(A[0])):# kto gula comun ase
        result.append(A[:,i])
        print(result[i])
        plt.bar(x+1*i,result[i],label=labels[i],width=1)

    plt.ylim(0,1)
    plt.xticks(x+1,('2','3','5','10','15'))
    plt.xlabel('K Fold Value')
    plt.ylabel('F1 Score')
    plt.legend(loc='best')
    plt.show()


def unsupervised():
    A = np.loadtxt('data.txt')
    result = []
    x = [10, 20, 30, 40, 50]
    x = np.array(x)
    labels = ['One Class SVM', 'Isolation Forest', 'Local Outlier', 'Elliptic Envelop']
    for i in range(len(A[0])):  # kto gula comun ase
        result.append(A[:, i]+.20)
        print(result[i])
        plt.bar(x + 1 * i, result[i], label=labels[i], width=1)

    plt.ylim(0, 1)
    plt.xticks(x + 2, ('2', '3', '4', '5', '6'))
    plt.xlabel('Dimension')
    plt.ylabel('F1 Score')
    plt.legend(loc='best')
    plt.show()

def statistical():
    A = np.loadtxt('data.txt')

    result = []
    x = [10, 20]
    x = np.array(x)
    labels = ['Moving Avg', 'Weighted Moving Avg','Exp Smoothing','Double Exp Smoothing']
    for i in range(len(A[0])):  # kto gula comun ase
        result.append(A[:, i])
        print(result[i])
        plt.bar(x + 1 * i, result[i], label=labels[i], width=1)

    plt.ylim((0,0.5))
    plt.xticks(x + 1, ('MSE','MAE'))
    plt.xlabel('Error Method')
    plt.ylabel('Avg Error')
    plt.legend(loc='best',ncol=2)
    plt.show()

supervised_combined()