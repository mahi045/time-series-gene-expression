import matplotlib.pyplot as plt
import numpy as np

A = np.loadtxt('mptoh.txt')

result = []
x = [10,20,30,40,50]
x = np.array(x)
plt.figure().suptitle('Disk no fixed, n= 20')
labels = ['Baseline','Efficient']
for i in range(2):# jto gula comun ase
    result = A[i]
    plt.bar(x+i*2,result,label=labels[i],width=2)

plt.xticks(x+1,('4','5','6','7','8'))
plt.xlabel('Number of pegs')
plt.ylabel('Number of Moves')
plt.legend(loc='best')
plt.show()