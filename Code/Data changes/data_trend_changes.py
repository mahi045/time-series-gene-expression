import matplotlib.pyplot as plt
import numpy as np
import random


f = open('GSE1723_series_matrix.txt')
intotal = list()
count = 0
for line in f:
    if count >= 200:
        break
    line = line.split()
    data = line[2:]
    data = [float(_) for _ in data]
    data = np.array(data)
    data = data - data.min()
    data = data / data.max()
    if line[1] == '1':
        count += 1
        intotal.append(data.tolist())
        

selected = random.sample(intotal, 5)     
for _ in selected:
    plt.plot(_) 
plt.show()