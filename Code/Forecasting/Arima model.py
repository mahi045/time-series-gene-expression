import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import  ARIMA
from Util import create_dataset, create_GSE_dataset
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
DATASET = 'GSE20305_series_matrix.txt' 
DATA = create_GSE_dataset(DATASET)
N_GENE = len(DATA)
N_TIME = len(DATA[0])
TEST_PERCENT = 0.3
LAST_IDX = int(np.ceil(N_TIME*(1-TEST_PERCENT)))


def arima_model(index):
    series = DATA[index]
    print('GENE number ', index)
    train = series[:LAST_IDX]
    test = series[LAST_IDX:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(3, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    error =sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % error)
    predicted_series = train + predictions
    # fig = plt.figure()
    # #fig.suptitle(str(index+1)+' th gene')
    # plt.plot(predicted_series,c='r',label = 'predicted')
    # plt.plot(series, c='g',label = 'actual')
    # plt.legend(loc = 'best')
    # plt.draw()
    return error


def main():

    global TEST_PERCENT,LAST_IDX
    f = open('out.txt', 'a')
    f.write(DATASET + ' - Arima\n')
    for TEST_PERCENT in [0.3]:
        LAST_IDX = int(np.ceil(N_TIME * (1 - TEST_PERCENT)))
        RMSE_ERROR = list()
        sample_size = 100
        for i in range(sample_size):
            RMSE_ERROR.append(arima_model(i))
            # print('Avf ERRor so far ',np.average(RMSE_ERROR))
        print('AVG MSE ERROR', np.average(RMSE_ERROR))
        f.write('test percent ' + str(TEST_PERCENT))
        f.write(' avg rmse ' + str(np.average(RMSE_ERROR)))
        f.write('\n')
    plt.show()


if __name__ == '__main__':
    main()