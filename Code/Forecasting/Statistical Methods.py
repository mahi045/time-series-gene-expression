import matplotlib.pyplot as plt
from Util import create_dataset
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from anompy.detector.smoothing import TripleExponentialSmoothing


DATA = create_dataset()
N_GENE = len(DATA)
N_TIME = len(DATA[0])
TEST_PERCENT = 0.3
LAST_IDX = 0


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen


def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals


def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result


def holtz_winter(series):

    forecasted_series = triple_exponential_smoothing(series=series[:LAST_IDX],slen=18,alpha=0.5,beta=0.1,gamma=0.65,n_preds=N_TIME-LAST_IDX)
    forecasted_series = series[:LAST_IDX] + forecasted_series[LAST_IDX:]
    rmse = sqrt(mean_squared_error(series[LAST_IDX:],forecasted_series[LAST_IDX:]))
   # print(rmse)
    # plt.figure().suptitle('')
    # plt.plot(forecasted_series,color='red',label='forecast')
    # plt.plot(series, color='green', label='actual')
    # plt.legend(loc='best')
    return rmse


#holtz_winter(DATA[1])
def main():
    global TEST_PERCENT
    global LAST_IDX

    f = open('out.txt','a')
    f.write('Holtz Winter\n')
    for TEST_PERCENT in [0.1,0.2,0.3,0.4]:
        LAST_IDX = int(np.ceil(N_TIME * (1 - TEST_PERCENT)))
        rmse_list = np.zeros(N_GENE)
        for i in range(N_GENE):
            rmse_list[i] = holtz_winter(DATA[i])

        print('AVG RMSE: ',np.average(rmse_list))
        f.write('test percent '+str(TEST_PERCENT))
        f.write(' avg rmse '+ str(np.average(rmse_list)))
        f.write('\n')
    plt.show()

if __name__ == '__main__':
    main()