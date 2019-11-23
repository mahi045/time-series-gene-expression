import anompy
import matplotlib.pyplot as plt
from anompy.detector.base import BaseDetector
from anompy.detector.average import AverageDetector
from anompy.detector.smoothing import ExponentialSmoothing, DoubleExponentialSmoothing, TripleExponentialSmoothing
import numpy as np
from Util import create_dataset
from sklearn.metrics import mean_squared_error,mean_absolute_error

THREESHOLD = 0.7  # 0,7 for basic data, 15 for ccycle
SCALE = 1

# actual list of float, forecasted list of (float,bool) tuple
def plot_series(actual,forecasted):
    plt.figure().suptitle('single')
    markers_on = []
    for i in range(len(actual)-1):
        if forecasted[i][1] : # annomolous point
            plt.plot(actual[i],color = 'b',marker = '*')
        else:
            plt.plot(actual[i], color='b', marker='o')

    for i in range(len(forecasted)):
        plt.plot(forecasted[i][0],color = 'r')
    plt.draw()

def get_anomolous_points(actual,forecasted):
    anomolous_points = []
    for i in range(len(actual) - 1):
        if forecasted[i][1]:  # annomolous point
            anomolous_points.append(i)
    return anomolous_points


def base_detector(data):
    for series in data:
        detector = BaseDetector(series[0],threshold=0.5)
        forecasted_series = detector.detect(series[1:])
        #print(forecasted_series)
        base_series = [t[0] for t in forecasted_series] + [detector.observed_last]
        print(series)
        print(base_series)
        '''
        plt.figure().suptitle('Single Data')
        plt.plot(base_series)
        #print(np.array(forecasted_series)[:,0])
        plt.plot(np.array(forecasted_series)[:,0])
        print(len(forecasted_series))
        print(len(base_series))
        for i in range(len(base_series) - 1):
            print(base_series[i],forecasted_series[i])
        plt.draw()
        '''''
        break

    def avg_detector(series):

        #print(THREESHOLD)
        detector = AverageDetector(series[0],window_size=3,threshold=THREESHOLD)
        forecasted_series = detector.detect(series[1:])
        simple_average_series = [t[0] for t in forecasted_series] + [detector.average]

        # for i in range(len(series) - 1):
        #     print(series[i], forecasted_series[i])

        anomolous_points = get_anomolous_points(series,forecasted_series)
        print(anomolous_points)
        # fig = plt.figure()
        # fig.suptitle('')
        # plt.plot(series, '-D', markevery=anomolous_points, label='anomolous point')
        # plt.plot(series, label='actual')
        # plt.plot(simple_average_series, label='moving avg',alpha=2)
        # plt.legend(loc='best')
        # plt.xlabel('time')
        # plt.ylabel('ratio')
        # #fig.savefig('fig/ccycle_moving_avg.png')
        # plt.draw()
        mse = mean_squared_error(series,simple_average_series)
        mae = mean_absolute_error(series,simple_average_series)
        #print(mse)
        return anomolous_points,mse,mae


def weighted_avg_detector(series):
        weights = [.2,.3,.5]
        detector = AverageDetector(series[0],window_size=3,threshold=THREESHOLD,weights=weights)
        forecasted_series = detector.detect(series[1:])
        weighted_average_series = [t[0] for t in forecasted_series] + [detector.average]
        # for i in range(len(series) - 1):
        #     print(series[i], forecasted_series[i])
        anomolous_points = get_anomolous_points(series,forecasted_series)
        print(anomolous_points)
        # fig = plt.figure()
        # fig.suptitle('')
        # plt.plot(series, '-D', markevery=anomolous_points, label='anomolous point')
        # plt.plot(series, label='actual')
        # plt.plot(weighted_average_series, label='weighted avg',alpha=2)
        # plt.legend(loc='best')
        # plt.xlabel('time')
        # plt.ylabel('ratio')
        # #fig.savefig('fig/ccycle_weighted_avg.png')
        # plt.draw()
        mse = mean_squared_error(series, weighted_average_series)
        mae = mean_absolute_error(series,weighted_average_series)
        #print(mse)
        return anomolous_points,mse,mae



def exponential_smoothing(series):
        weights = [.2,.3,.5]
        detector = ExponentialSmoothing(series[0], alpha=0.5,threshold=THREESHOLD)
        forecasted_series = detector.detect(series[1:])
        smoothing_series = [t[0] for t in forecasted_series] + [detector.forecasted]
        # for i in range(len(series) - 1):
        #     print(series[i], forecasted_series[i])
        anomolous_points = get_anomolous_points(series,forecasted_series)
        print(anomolous_points)
        # fig = plt.figure()
        # fig.suptitle('')
        # plt.plot(series, '-D', markevery=anomolous_points, label='anomolous point')
        # plt.plot(series, label='actual')
        # plt.plot(smoothing_series, label='exp smoothing',alpha=2)
        # plt.legend(loc='best')
        # plt.xlabel('time')
        # plt.ylabel('ratio')
        # #fig.savefig('fig/ccycle_exp.png')
        # plt.draw()
        mse = mean_squared_error(series, smoothing_series)
        mae = mean_absolute_error(series,smoothing_series)
        #print(mse)
        return anomolous_points,mse,mae


def double_exp_smoothing(series):
    detector = DoubleExponentialSmoothing(series[0], alpha=0.8, beta=0.6,threshold=THREESHOLD)
    forecasted_series = detector.detect(series[1:])
    double_smoothing_series = [t[0] for t in forecasted_series] + [detector.forecasted]
    anomolous_points = get_anomolous_points(series, forecasted_series)
    print(anomolous_points)
    # fig = plt.figure()
    # fig.suptitle('')
    # plt.plot(series, '-D', markevery=anomolous_points,label = 'anomolous point')
    # plt.plot(series,  label='actual')
    # plt.plot(double_smoothing_series, label='double exp smoothing',alpha=2)
    # plt.legend(loc='best')
    # plt.xlabel('time')
    # plt.ylabel('ratio')
    # #fig.savefig('fig/ccycle_double_exp.png')
    # plt.draw()
    mse = mean_squared_error(series, double_smoothing_series)
    mae = mean_absolute_error(series,double_smoothing_series)
    #print(mse)
    return anomolous_points,mse,mae


def triple_exp_smoothing(series):
    detector = TripleExponentialSmoothing(series[0:1], season_length=3, alpha=0.3, beta=0.1, gamma=0.9,threshold=0.7)
    forecasted_series = detector.detect(series[1:])
    triple_smoothing_series = [t[0] for t in forecasted_series] + [detector.forecasted]
    anomolous_points = get_anomolous_points(series, forecasted_series)
    print(anomolous_points)
    plt.figure().suptitle(' data point in triple exponential smoothing')
    plt.plot(series, '-D', markevery=anomolous_points)
    plt.plot(triple_smoothing_series)
    plt.draw()


def create_labelled_annomaly(data):
    label = np.zeros((len(data),len(data[0])))   # num_gene * num_point

    # print(np.shape(label))
    AVG_MSE = np.zeros(shape=(len(data),4))
    AVG_MAE = np.zeros(shape=(len(data),4))
    for i in range(len(data)):
        print(i,'th gene')
        a1,AVG_MSE[i][0],AVG_MAE[i][0] = avg_detector(data[i])
        a2,AVG_MSE[i][1],AVG_MAE[i][1] = weighted_avg_detector(data[i])
        a3,AVG_MSE[i][2],AVG_MAE[i][2] = exponential_smoothing(data[i])
        a4,AVG_MSE[i][3],AVG_MAE[i][3] = double_exp_smoothing(data[i])
        #print(a1,a2,a3,a4)
        for j in range(len(data[i])):
            if j in a1: label[i][j] += 1 / 4
            if j in a2: label[i][j] += 1 / 4
            if j in a3: label[i][j] += 1 / 4
            if j in a4: label[i][j] += 1 / 4
        print(label[i])
    print('AVG mse',AVG_MSE.mean(axis=0))
    print('AVG mae',AVG_MAE.mean(axis=0))
    np.savetxt('yeast_label.txt',label,fmt='%-7.2f')


def create_label_annomaly_changing_threshold(data):
    label = np.zeros((len(data), len(data[0])))  # num_gene * num_point

    # print(np.shape(label))
    AVG_MSE = np.zeros(shape=(len(data), 4))
    AVG_MAE = np.zeros(shape=(len(data), 4))
    for i in range(len(data)):
        print(i, 'th gene')
        global THREESHOLD
        THREESHOLD = 1
        iteration = 0
        while iteration<10:
            #print('threshold:',THREESHOLD)
            a1, AVG_MSE[i][0], AVG_MAE[i][0] = avg_detector(data[i])
            a2, AVG_MSE[i][1], AVG_MAE[i][1] = weighted_avg_detector(data[i])
            a3, AVG_MSE[i][2], AVG_MAE[i][2] = exponential_smoothing(data[i])
            a4, AVG_MSE[i][3], AVG_MAE[i][3] = double_exp_smoothing(data[i])
            iteration +=1
            if len(a4) < 4:
                THREESHOLD -= 0.1
            elif len(a4) > 15:
                THREESHOLD += 0.1
            else:
                break
        # print(a1,a2,a3,a4)
        for j in range(len(data[i])):
            if j in a1: label[i][j] += 1 / 4
            if j in a2: label[i][j] += 1 / 4
            if j in a3: label[i][j] += 1 / 4
            if j in a4: label[i][j] += 1 / 4
        print(THREESHOLD, label[i])
    print('AVG mse', AVG_MSE.mean(axis=0))
    print('AVG mae', AVG_MAE.mean(axis=0))
    #np.savetxt('ccycle_label.txt',label,fmt='%-7.2f')




def main():
    data = create_dataset()
    #base_detector(data)
    i = 0
    #create_label_annomaly_changing_threshold(data)    #for ccycle
    create_labelled_annomaly(data)     # for basic and yeast
    # label = np.loadtxt('basic_label.txt')
    # print(label)
    # avg_detector(data[i])
    # weighted_avg_detector(data[i])
    # exponential_smoothing(data[i])
    # double_exp_smoothing(data[i])
    #triple_exp_smoothing(data[i])
    plt.show()


if __name__ == '__main__':
    main()