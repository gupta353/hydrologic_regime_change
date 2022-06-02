import numpy as np
import  statsmodels.api as sm
import FarimaModule
from  statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# function to compute running mean
def runningMean(data, w):
    x = data.copy()
    w2 = round(w/2)
    xm = []
    for ind in range(w2, len(x)-w2):
        xm.append(np.mean(x[ind-w2:ind+w2]))
    return xm
#####################################################################################################

np.random.seed(10)

# FARIMA model
p = 1                               # autoregressive order
q = 1                              # moving-average order
arparams = np.array([0.25])    # autoregressive coefficients
maparams = np.array([0.5])        # moving-average coefficients
ar = np.r_[1, -arparams]
ma = np.r_[1, maparams]
sigma = 4                         # varaince of white noise
N = 10000                         # number of time-steps at which data will be generated
x = np.arange(1,N+1,1)

# generate and plot data
fig, ax = plt.subplots(3, 1, figsize = (15, 8))
plt.rc('font', **{'family' : 'Arial', 'size' : 12})

data = sm.tsa.arma_generate_sample(ar,ma,N,sigma)
ax[0].plot(x[0:200],data[0:200], linewidth = 2)

# apply inverse of the differencing operator for different values of d
d = 0.30
pt = 100
data1 = FarimaModule.invdiffOp(data,d,pt)
ax[0].plot(x[0:200],data1[0:200], '--', linewidth = 2)

# generate a time-series with different autoregressive parameter and d = 0
arparams = np.array([0.75])    # autoregressive coefficients
ar = np.r_[1, -arparams]
data2 = sm.tsa.arma_generate_sample(ar,ma,N,sigma)
ax[0].plot(x[0:200], data2[0:200], ':', linewidth = 2)

# compute running mean with window length 20 and plot
data_mean1 = runningMean(data, 30)
data1_mean1 = runningMean(data1, 30)
data2_mean1 = runningMean(data2, 30)

ax[1].plot(data_mean1[0:1000], linewidth = 2)
ax[1].plot(data1_mean1[0:1000], '--', linewidth = 2)
ax[1].plot(data2_mean1[0:1000], ':', linewidth = 2)

# compute running mean with window length 200 and plot
data_mean2 = runningMean(data, 365)
data1_mean2 = runningMean(data1, 365)
data2_mean2 = runningMean(data2, 365)

ax[2].plot(data_mean2, linewidth = 2)
ax[2].plot(data1_mean2, '--', linewidth = 2)
ax[2].plot(data2_mean2, ':', linewidth = 2)

ax[2].set_xlabel('Time (days)')
ax[0].set_ylabel('$X(t)$')
ax[1].set_ylabel('1-month moving\naveraged $X(t)$')
ax[2].set_ylabel('1-year moving\naveraged $X(t)$')
ax[0].legend(['$\phi_1 = 0.25, d = 0.00$', '$\phi_1 = 0.25, d = 0.30$', '$\phi_1 = 0.75, d = 0.00$'], frameon = False)

ax[0].text(0, 20, '(a)', weight = 'bold')
ax[1].text(0, 8, '(b)', weight = 'bold')
ax[2].text(0, 4, '(c)', weight = 'bold')

fig.tight_layout()

# save plot
save_direc = 'D:/Research/non_staitionarity/codes/results/miscellaneous_plots'
sname = 'Farima_time_series_different_parameters.png'
filename = save_direc + '/' + sname
plt.savefig(filename, dpi = 600)