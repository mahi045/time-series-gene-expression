from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import LSTM
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from Util import create_dataset, create_GSE_dataset

# DATA = create_dataset()
DATASET = 'GSE3406_series_matrix.txt' 
DATA = create_GSE_dataset(DATASET)
N_GENE = len(DATA)
N_TIME = len(DATA[0])
TEST_PERCENT = 0.3
LAST_IDX = int(np.ceil(N_TIME*(1-TEST_PERCENT)))
TEST_SIZE = N_TIME - LAST_IDX + 1


def normalise_series(series):
    base = series[0]
    for i in range(len(series)):
        series[i] /=base
    return series

def original_series(series,base):
    for i in range(len(series)):
        series[i] *=base
    return series


# ready dataset as input, output
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    #print(len(dataY))
    return np.array(dataX), np.array(dataY)


def lstm_method(series):
    # pre-process the data in (-1,1) scale
    # print(series)
    series = np.array(series)
    values = series.reshape(-1,1)
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)


    #split data
    train, test = scaled[0:LAST_IDX, :], scaled[LAST_IDX:len(scaled), :]

    #  Generate dataset for trainX, trainY, testX, testY
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape X for model training

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    #print(trainX.shape)
    # Create model
    model = Sequential()
    model.add(LSTM(100,return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    #model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=500, batch_size=100, validation_data=(testX, testY), verbose=0,
                        shuffle=False)

    #show loss
    # pyplot.figure().suptitle('loss in LSTM during training')
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend(loc='best')

    # Predict
    yhat = model.predict(testX)

    #scale back to original
    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))

    #my normaliser

    # pyplot.figure('Forecasting')
    # pyplot.plot(yhat_inverse, label='forecast', color='red')
    # pyplot.plot(testY_inverse, label='actual', color='green')
    # pyplot.legend(loc='best')
    #RMSE error
    rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
    print('Test RMSE: %.3f' % rmse)


    # pyplot.figure('')
    # yhat_inverse = list(series[0:LAST_IDX+1])+list(yhat_inverse.ravel())
    # pyplot.plot(yhat_inverse, label='forecast',color='red')
    # pyplot.plot(series, label='actual', color = 'green')
    # pyplot.legend(loc='best')
    return rmse


def main():
    global TEST_PERCENT,LAST_IDX,TEST_SIZE
    f = open('out.txt', 'a')
    #f.write('\nLSTM\n')
    global TEST_PERCENT
    for TEST_PERCENT in [ 0.4]:
        LAST_IDX = int(np.ceil(N_TIME * (1 - TEST_PERCENT)))
        TEST_SIZE = N_TIME - LAST_IDX + 1
        rmse_list = []
        sample_size = 100
        f.write('LSTM method - %s\n' %(DATASET))
        for i in range(sample_size):
            print('LSTM GENE no',i)
            #DATA[i] = normalise_series(DATA[i])
            rmse_list.append(lstm_method(DATA[i]))
            print('AVG so far, ', np.average(rmse_list))

        print('AVG RMSE: ',np.average(rmse_list))
        f.write('test percent ' + str(TEST_PERCENT))
        f.write(' avg rmse ' + str(np.average(rmse_list)))
        f.write('\n')
    pyplot.show()

if __name__ == '__main__':
    main()