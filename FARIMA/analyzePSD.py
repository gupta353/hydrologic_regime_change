"""
This script is written to analyze the effect of FARIAMA model parameters on PSD

Author: Abhinav Gupta (Created: 12 Nov 2021)

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import optimize
from scipy.optimize.nonlin import Jacobian
import  statsmodels.api as sm
import FarimaModule
from  statsmodels.tsa.arima.model import ARIMA
from scipy.integrate import simpson

np.random.seed(10)

p = 1                              # autoregressive order
q = 1                              # moving-average order
arparams = np.array([0.10])    # autoregressive coefficients
maparams = np.array([0.1])        # moving-average coefficients
ar = np.r_[1, -arparams]
ma = np.r_[1, maparams]
sigma = 1                        # varaince of white noise
N = 3650                            # number of time-steps at which data will be generated
x = np.arange(1,N+1,1)

"""
data = sm.tsa.arma_generate_sample(ar,ma,N,sigma)
plt.plot(x,data)


# apply inverse of the differencing operator
if (d != 0):
    pt = 100
    data = FarimaModule.invdiffOp(data,d,pt)
plt.plot(x,data)
plt.show()
"""

# frequencies at which PSD is to be computed
fs = 1
fmax = fs/2
fint = 2*fmax/N
f = np.arange(0,fmax,fint)
f = f[1:]

# Effetc of d 
d_list = [0.10,0.20,0.30,0.40,0.50]
legend_labels = ['$\itd$ = ' + str(round(x*100)/100) for x in d_list]
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['lines.markersize'] = 1
linestyles = ['-','--','-.',':', 'o']
fig, ax = plt.subplots(1,3, figsize=(15,5))
count = -1
for d in d_list:
    count = count + 1
    pxx_den = FarimaModule.pxx_denFARIMA(p,d,q,ar,ma,f,sigma,True)
    sigmax2 = simpson(pxx_den, x=2*np.pi*f, even='avg')
    pxx_den = pxx_den/sigmax2
    ax[0].loglog(f,pxx_den, linestyles[count])
    ax[0].set_xlabel('$\itf$ (Cycles per day)')
    ax[0].set_ylabel('$\itf_X(2\pi\itf)$')
    ax[0].legend(legend_labels, frameon = False)
    ax[0].grid(linestyle = '--')

# effect of AR parameters
ar_list = list(np.arange(0,1.0,0.20))
legend_labels = ['$\it\phi_1$ = ' + str(round(x*100)/100) for x in ar_list]
d = 0.1
count = -1
for ar1 in ar_list:
    count = count + 1
    arparams = np.array([ar1])    # autoregressive coefficients
    ar = np.r_[1, -arparams]
    pxx_den = FarimaModule.pxx_denFARIMA(p,d,q,ar,ma,f,sigma,True)
    sigmax2 = simpson(pxx_den, x=2*np.pi*f, even='avg')
    pxx_den = pxx_den/sigmax2
    ax[1].loglog(f,pxx_den, linestyles[count])
    ax[1].set_xlabel('$\itf$ (Cycles per day)')
    ax[1].set_ylabel('$\itf_X(2\pi\itf)$')
    ax[1].legend(legend_labels, frameon = False)
    ax[1].set_ylim(0.02, 10**2)
    ax[1].grid(linestyle = '--')

# effect of MA parameters
d = 0.1
arparams = np.array([0.10])    # autoregressive coefficients
ar = np.r_[1, -arparams]
ma_list = list(np.arange(0.1,1.0,0.2))
legend_labels = ['$\it\psi_1$ = ' + str(round(x*100)/100) for x in ma_list]
count = -1
for ma1 in ma_list:
    count = count + 1
    maparams = np.array([ma1])    # autoregressive coefficients
    ma = np.r_[1, maparams]
    pxx_den = FarimaModule.pxx_denFARIMA(p,d,q,ar,ma,f,sigma,True)
    sigmax2 = simpson(pxx_den, x=2*np.pi*f, even='avg')
    pxx_den = pxx_den/sigmax2
    ax[2].loglog(f,pxx_den,linestyles[count])
    ax[2].set_xlabel('$\itf$ (Cycles per day)')
    ax[2].set_ylabel('$\itf_X(2\pi\itf)$')
    ax[2].legend(legend_labels, frameon = False)
    ax[2].set_ylim(0.02, 10**2)
    ax[2].grid(linestyle = '--')
fig.tight_layout()

ax[0].text(0.0003, 100, '(a)', weight = 'bold')
ax[1].text(0.0003, 30, '(b)', weight = 'bold')
ax[2].text(0.0003, 30, '(c)', weight = 'bold')

# save plot
sname = 'psd_vs_FARIMA_parameters.png'
save_direc = 'D:/Research/non_staitionarity/codes/results/miscellaneous_plots'
filename = save_direc + '/' + sname
plt.savefig(filename, dpi = 300)

plt.show()