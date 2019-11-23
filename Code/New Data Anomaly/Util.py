import numpy as np
import pandas as pd

def get_patient_data(normalise=True):   # return gene name, class and values

    filename = 'patient_data.txt'
    f = open(filename,'r')
    gene_names = []
    gene_types = []
    data = []
    for line in f:
        lst = line.split()
        gene_names.append(lst[0])
        gene_types.append(lst[1])
        a = []
        #print(lst[1])
        for val in lst[2:]:
            a.append(float(val))
        data.append(a)
    data = np.array(data)
    # print(data.shape)
    # print(data.max(axis=0))
    if normalise:
        data = data / data.max(axis=0)
    # print(data)
    return gene_names,gene_types,data


def one_hot(labels, n_class=2):
    y = [0]*len(labels)
    for i in range(len(labels)):
        if labels[i] == 'good':
            y[i] = [1, 0]
        else:             # bad
            y[i] = [0,1]
    return y

def get_batches(X, y, batch_size = 100):
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]
    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size].reshape(batch_size,9,1), y[b:b + batch_size].reshape(batch_size,1)


def get_yeast_data(filename,normalise=False):
    #filename = 'patient_data.txt'
    f = open(filename, 'r')
    gene_names = []
    gene_types = []
    data = []
    for line in f:
        lst = line.split()
        gene_names.append(lst[0])
        gene_types.append(lst[1])
        a = []
        # print(lst[1])
        for val in lst[2:]:
            a.append(float(val))
        data.append(a)
    data = np.array(data)
    # print(data.shape)
    # print(data.max(axis=0))
    if normalise:
        data = data / data.max(axis=0)
    # print(data)
    return gene_names, gene_types, data

