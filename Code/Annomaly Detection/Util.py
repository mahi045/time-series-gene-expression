import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
DATASET = 'ccycle.txt'
#DATASET = 'basic.txt'
#DATASET = 'yeast.txt'

def create_dataset():
    f = open(DATASET,'r')
    f.readline()
    data = []
    # fig = plt.figure()
    # fig.suptitle('all_points')
    for line in f:
        lst = line.split()
        lst = lst[1:]
        lst = [float(i) for i in lst]
        # if DATASET =='ccycle.txt' :
        #     lst = normalise_row(lst)
        # #plt.plot(lst)
        #print(lst)
        data.append(lst)
    #plt.show()
    # print('drawn')
    f.close()
    np.savetxt('ccycle_modified.txt',data,fmt='%-7.2f')
    return data



def create_yeast_data():
    #data = pd.read_csv('yeast.txt', sep=" ", header=None)
    f = open('yeast_data.txt','r')
    fw = open('yeast.txt','w')
    len_list = []
    for line in f:
        lst = line.split()
        actual_list = []
        for item in lst:
            try:
                v = float(item)
                if v<10:
                    actual_list.append(v)
            except:
                continue
        while len(actual_list)<175:
            v = actual_list[-1] + random.uniform(-0.2,0.2)
            actual_list.append(v)
        len_list.append(len(actual_list))
        print(len(line),len(actual_list),actual_list)
        for item in actual_list:
            fw.write(str(round(item,2))+'   ')
        fw.write('\n')
    #print(min(len_list),max(len_list))


def normalise_row(row):
    base = row[0]
    row = [val/base for val in row]
    return row

def binary_convertor_annomaly_label(label):
    for series in label:
        for i in range(len(series)):
            if series[i]>=0.5:
                series[i] = 1
            else:
                series[i] = 0
    return label

create_dataset()
#create_yeast_data()