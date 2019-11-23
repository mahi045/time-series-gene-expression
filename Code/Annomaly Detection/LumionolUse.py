from luminol.anomaly_detector import AnomalyDetector
from luminol.correlator import Correlator
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# return a list of dictionary of time seies gene data
def create_dataset():
    f = open('basic.txt','r')
    f.readline()
    data = []
    for line in f:
        lst = line.split()
        lst = lst[1:]
        lst = [float(i) for i in lst]
        #plt.plot(lst)
        #print(lst)

        d = dict(enumerate(lst))
        data.append(d)

    f.close()
    return data

# bitmap_detector is the anomaly detector algorithm
def get_annomaly_in_individual_series(data):
    plt.figure()
    for ts in data:
        index = data.index(ts)
        my_detector = AnomalyDetector(ts,algorithm_name='default_detector')
        score = my_detector.get_all_scores()
        anomalies = my_detector.get_anomalies()
        print(score)
        X =[]   # annomaly point
        Y = []  # annomaly score
        size = []
        for a in anomalies:
            print(a.get_time_window())
            #print(a.exact_timestamp)
            for time in range(a.start_timestamp,a.end_timestamp+1):
                X.append(time)
                Y.append(a.anomaly_score)
                size.append(a.anomaly_score*50)
        print(X,Y)
        plt.scatter(X,Y,s=size,label = str(index),marker='o' )
        #plt.plot(X)
        plt.legend(loc='best',ncol = 5)

def get_corelations(data):
    num_of_data = len(data)
    correlation_matrix = np.ones((num_of_data,num_of_data))
    for i in range(num_of_data):
        for j in range(num_of_data):
            my_correaltor = Correlator(data[i],data[j])
            correlation_result = my_correaltor.get_correlation_result()
            #print(correlation_result.coefficient)
            correlation_matrix[i][j] = correlation_result.coefficient
            correlation_matrix[j][i] = correlation_result.coefficient
    #cor = np.corrcoef(list(data[0].values()),list(data[1].values()))
    print(correlation_matrix)
    np.fill_diagonal(correlation_matrix, 1)
    dissimilarity = 1 - np.abs(correlation_matrix)
    hierarchy = linkage(squareform(dissimilarity), method='average')
    labels = fcluster(hierarchy, 0.9, criterion='distance')
    print(labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlation_matrix, vmin=-1, vmax=1)
    fig.colorbar(cax)
    plt.show()

data = create_dataset()
get_annomaly_in_individual_series(data)
#get_corelations(data)
#plt.show()
'''
#ts = {0: 0, 1: 0.5, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0}

my_detector = AnomalyDetector(ts)
score = my_detector.get_all_scores()
for timestamp, value in score.iteritems():
    print(timestamp, value)

'''