import matplotlib.pyplot as plt
import numpy as np

A = np.loadtxt('data.txt')

result = []
x = [10, 20,30,40]
x = np.array(x)
labels = ['Holtz Winter','Arima model','Artificial Neural Network','LSTM method']
for i in range(len(A[0])):# jto gula comun ase
    result.append(A[:,i])
    print(result[i])
    plt.bar(x+1*i,result[i],label=labels[i],width=1)

plt.xticks(x+2,('10','20','30','40'))
plt.xlabel('Test percent')
plt.ylabel('Root mean square erro(AVG)')
plt.legend(loc='best')
plt.show()