"""
This script computes difference in nse obtained by the RF models with different sets of predictor variables

Author: Abhinav Gupta (Created: 29 Apr 2022)

"""

import os
import matplotlib.pyplot as plt
import numpy as np

#
def readNSE(filename):
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    nse = float(data[0].split('=')[1]) 
    return nse

direc_1 = 'D:/Research/non_staitionarity/codes/results/RF_predictions_in_time_2'
direc_2 = 'D:/Research/non_staitionarity/codes/results/RF_predictions_in_time_3'

listFile = os.listdir(direc_1)
listFile = [fname for fname in listFile if fname.endswith('.txt')]

nse1 = []
nse2 = []
for fname in listFile:
    
    # read data for the first set of predictor variables
    filename = direc_1 + '/' + fname
    nse1.append(readNSE(filename))

    filename = direc_2 + '/' + fname
    nse2.append(readNSE(filename))

nse1 = np.array(nse1)
nse2 = np.array(nse2)
diff = nse2 - nse1

a = 1
plt.scatter(nse1, nse2)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], color = 'black')
plt.show()



