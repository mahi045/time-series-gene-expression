import matplotlib.pyplot as plt
from anompy.detector.base import BaseDetector
from anompy.detector.average import AverageDetector
from anompy.detector.smoothing import ExponentialSmoothing, DoubleExponentialSmoothing, TripleExponentialSmoothing
import numpy as np
from Util import get_patient_data
import matplotlib.pyplot as plt

THREESHOLD = 0.2
NUM_POINTS = 9
time = [1,2,3,4,5,6,7,8,9]

def get_anomolous_points(actual,forecasted):
    anomolous_points = []
    for i in range(len(actual) - 1):
        if forecasted[i][1]:  # annomolous point
            anomolous_points.append(i)
    return anomolous_points

def avg_detector(series):

        #print(THREESHOLD)
    detector = AverageDetector(series[0],window_size=3,threshold=THREESHOLD)
    forecasted_series = detector.detect(series[1:])
    simple_average_series = [t[0] for t in forecasted_series] + [detector.average]

    # for i in range(len(series) - 1):
    #     print(series[i], forecasted_series[i])

    anomolous_points = get_anomolous_points(series,forecasted_series)
    return anomolous_points

def weighted_avg_detector(series):
        weights = [.2,.3,.5]
        detector = AverageDetector(series[0],window_size=3,threshold=THREESHOLD,weights=weights)
        forecasted_series = detector.detect(series[1:])
        weighted_average_series = [t[0] for t in forecasted_series] + [detector.average]
        # for i in range(len(series) - 1):
        #     print(series[i], forecasted_series[i])
        anomolous_points = get_anomolous_points(series,forecasted_series)
        return anomolous_points

def anomaly_detection(data,gene_type):
    # for i in range(len(data)):
    #     if gene_type[i] == 'bad':
    #         plt.plot(time,data[i])
    #
    # plt.show()
    good_points = np.zeros(NUM_POINTS)
    bad_points = np.zeros(NUM_POINTS)
    for i in range(len(data)):
        series = data[i]
        anomolous_points = weighted_avg_detector(series)
        print(gene_type[i],anomolous_points)
        for point in anomolous_points:
            if gene_type[i] == 'good':
                good_points[point] += 1
            else:
                bad_points[point] += 1
    print(good_points)
    print(bad_points)

def main():
    gene_names,gene_types,data = get_patient_data()
    anomaly_detection(data,gene_types)
    pass


if __name__ == '__main__':
    main()