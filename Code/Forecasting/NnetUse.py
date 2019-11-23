from nnet_ts import *
from Util import create_dataset, create_GSE_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
DATASET = 'GSE6186_series_matrix.txt' 
DATA = create_GSE_dataset(DATASET)
N_GENE = len(DATA)
N_TIME = len(DATA[0])
TEST_PERCENT = 0.3
LAST_IDX = int(np.ceil(N_TIME*(1-TEST_PERCENT)))


def nnet_use(index):
    time_series = np.array(DATA[index][0:LAST_IDX])
    time_series = np.exp(time_series)
    #print(time_series)
    neural_net = TimeSeriesNnet(hidden_layers=[5, 10, 5, 10],
                                activation_functions=['sigmoid', 'relu', 'relu', 'sigmoid'])
    neural_net.fit(time_series, epochs=10000)
    neural_net.predict_ahead(n_ahead=N_TIME-LAST_IDX)
    predicted_series = np.log(neural_net.timeseries)
    #predicted_series = neural_net.timeseries
    # print(predicted_series)
    # print(DATA[index])
    # plt.plot(range(len(neural_net.timeseries)), np.log(neural_net.timeseries), '-r', label='Predictions', linewidth=1)
    # plt.plot(DATA[index], '-g', label='Original series')
    # plt.show()
    rmse = sqrt(mean_squared_error(DATA[index][LAST_IDX:],predicted_series[LAST_IDX:]))
    print('Gene No',index,'rmse:',rmse)

    # plt.plot(predicted_series,c='r',label='forecast')
    # plt.plot(DATA[index], c='g', label='actual')
    # plt.legend(loc='best')
    return rmse



def main():
    global TEST_PERCENT,LAST_IDX
    # f = open('outANN.txt', 'a')
    # f.write('\nANN\n')
    for TEST_PERCENT in [0.4]:
        LAST_IDX = int(np.ceil(N_TIME * (1 - TEST_PERCENT)))
        rmse_list = list()
        sample_size = 100
        for i in range(sample_size):
            rmse_list.append(nnet_use(i))
            print('avg so far, ',np.average(rmse_list))

        print('AVG RMSE: ', np.average(rmse_list))
        print('--------------------------------------')
        # f.write('test percent ' + str(TEST_PERCENT))
        # f.write(' avg rmse ' + str(np.average(rmse_list)))
        # f.write('\n')
    plt.show()


if __name__ == '__main__':
    main()




