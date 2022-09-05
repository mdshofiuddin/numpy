import math
import sys
import numpy as np
from numpy import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from math import log
import pandas as pd
import scipy

a = np.array(45)
d = np.array([42,2,12,5],ndmin=5)
b = np.array([[42,2,12,5,4],[5,4,5,89,6]])
c = np.array([[[42,2,12,5,4],[5,4,5,89,6],[4,56,6,46,4],[4,56,6,46,4]]])
e = np.array([1,2,3,4,5,6,7,8,9,10,12,1,3,14,15,16])
arr = e.reshape(2,4,2)

# sb = sb.displot([0,1,2,3,4,5,6,8])
# plt.show() 

# print(np.__version__)

# print('Testing',d[5,0]) 
# print(a.ndim)
# print(d.ndim)
# print('Second element of first row: ',b[1,3])
# print('testing in c: ',c[0,3,3])
# print(c.ndim)

# print(b[0:2,2]) #slice 2 index in two arrays
# print(arr)
# print(type(arr))
# print(math.pi)
# print(random.choice([356876456446,54648543654,64254346464,4564646464364]))

# print(pd.__version__)




df = pd.read_csv('DataTables.csv')
# print(df)
# print(df.info())
# print(scipy.__version__)
print(matplotlib.__version__)

ypoints = np.array([3, 7, 1, 9])

plt.plot(ypoints, marker = 'o', ms = 10, mec = 'b')
plt.show()

#Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()