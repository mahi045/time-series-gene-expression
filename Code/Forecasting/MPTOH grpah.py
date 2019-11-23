import matplotlib.pyplot as plt
import numpy as np

A = np.loadtxt('mptoh.txt')

result = []
x = [10,20,30,40,50]
x = np.array(x)
plt.figure().suptitle('Peg no fixed, p= 5')
labels = ['Naive Approach','Our Approach']
for i in range(2):# jto gula comun ase
    result = A[i]
    plt.bar(x+i*2,result,label=labels[i],width=2)

plt.xticks(x+1,('5','10','15','20','25'))
plt.xlabel('Number of disks')
plt.ylabel('Number of Moves')
plt.legend(loc='best')
plt.show()