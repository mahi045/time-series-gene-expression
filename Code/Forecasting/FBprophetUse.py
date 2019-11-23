from fbprophet import Prophet
import numpy as np
import pandas as pd
from  Util import create_dataset
from datetime import datetime
from matplotlib import pyplot as plt

DATA = create_dataset()
N_GENE = len(DATA)
N_TIME = len(DATA[0])
TEST_PERCENT = 0.4
LAST_IDX = int(np.ceil(N_TIME*(1-TEST_PERCENT)))

#for fbptophet data need to be presented for each day format
def convert_index_array(lst):
    array = []
    array.append(['ds','y'])
    for i in range(len(lst)):
        a = []
        m = int(i/30) + 1
        date = datetime(year=2018, month=1, day=i+1)
        a.append(date)
        a.append(lst[i])
        array.append(a)
    print(array)
    return array


data = create_dataset()
index = 7
table = convert_index_array(data[index][0:25])
headers = table.pop(0)
df = pd.DataFrame(table,columns=headers)
print(df)

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=15)
print(future.tail())
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
print(data[index][35:])
# forecasted_values = list(forecast['yhat'])
# print(len(forecasted_values))

estimated_values = list(df['y']) + list(np.log(list(forecast['yhat'])[25:]))
print(len(estimated_values),estimated_values)
print(data[index])
plt.plot(estimated_values,'-r',label = 'forecasted')
plt.plot(data[index],'-g',label = 'actual')
plt.legend()
plt.show()




