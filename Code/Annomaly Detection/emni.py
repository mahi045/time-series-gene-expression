import numpy as np

x = np.zeros(10)
idx = [1,4,5,9]
np.put(x,ind=idx,v=1)
print(x)