"""
This script test different methods for FARIMA parameter estimation

Author: Abhinav Gupta (Created: 13 Oct 2021)

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import optimize
from scipy.optimize.nonlin import Jacobian
import  statsmodels.api as sm
import FarimaModule
from  statsmodels.tsa.arima.model import ARIMA

np.random.seed(10)

# FARIMA model
p = 3                               # autoregressive order
q = 3                               # moving-average order
d = 0.25
arparams = np.array([0.25,0.10,0.20])    # autoregressive coefficients
maparams = np.array([0.50,0.60,0.30])        # moving-average coefficients
ar = np.r_[1, -arparams]
ma = np.r_[1, maparams]
sigma = 1.0                         # varaince of white noise
N = 3650                            # number of time-steps at which data will be generated
x = np.arange(1,N+1,1)

data = sm.tsa.arma_generate_sample(ar,ma,N,sigma)
plt.plot(data)
plt.show()

# apply inverse of the differencing operator
if (d != 0):
    pt = 100
    data = FarimaModule.invdiffOp(data,d,pt)
plt.plot(x,data)
plt.show()

# compute periodogram and remove zero frequency
fs = 1
f, pxx_den = FarimaModule.periodogramAG(data,fs,True)
f = f[1:]
pxx_den = pxx_den[1:]

#################################################################################################################################
# minimize the Whittle's approximate likelihood function using generalized simulated annealing 
""" def func(x):
    Lw = FarimaModule.objectiveFnc(x,p,q,f,pxx_den,True,False)
    return Lw

lw = [-1]*6 + [0]*2
up = [1]*6 + [0.5] + [2]

x0 = [0.4,0.4,0.2,0.5,0.5,0.5,0.45,0.2]

x0 = np.array(x0) + np.r_[np.random.normal(0,0.1,(6,)),np.random.normal(0,0.05,(2,))]
ret = optimize.dual_annealing(func, bounds=list(zip(lw, up)), maxiter = 5000, x0 = x0, initial_temp = 10000, visit = 3, no_local_search = True)
print(ret)

der = FarimaModule.derLw(p,q,ret.x,f,True,pxx_den)
print(der) """

#################################################################################################################################

# set derivatives to zero using fsolve method
""" one_sided = True
def funcder(x):
    fder = FarimaModule.derLw(p,q,x,f,one_sided,pxx_den)
    fder = np.array(fder)
    return abs(fder).tolist()

args = (p,q,f,True,pxx_den)
conv_test = False
theta = [0.5,0.5,0.5,0.5,0.5,0.5,0.2,0.5]
lw = [-1]*6 + [0]*2
up = [1]*6 + [0.5] + [2]
x0 = theta
count = 0
while ((conv_test == False) or (count<100)):
    count  = count + 1
    #roots, info, ier, mesg = optimize.fsolve(funcder, x0, args, full_output=True, maxfev = 10000 ,diag = [0.08]*8)
    roots = optimize.least_squares(funcder, x0, bounds = (tuple(lw),tuple(up)))
    val = funcder(roots.x)
    Lw = FarimaModule.objectiveFnc(roots.x,p,q,f,pxx_den,True,False)
    conv_test = np.isclose(val,[0]*8)
    conv_test = all(conv_test)
    print('solution :', roots.x)
    print('derivative :', val)
    print('-2*loglikeli :', Lw)
    x0 = theta + np.r_[np.random.normal(0,0.2,(6,)),np.random.normal(0,0.08,(1,)),np.random.normal(0,0.1,(1,))] """
#################################################################################################################################

# iterative d method with subsequent local minimization
one_sided = True
def funcder(x):
    fder = FarimaModule.derLw(p,q,x,f,one_sided,pxx_den)
    fder = np.array(fder)
    return abs(fder).tolist()
pt = 100
dvals = np.arange(0,0.5,0.01)
AIC_vals = []
for d in dvals:
    if (d != 0):
        y = FarimaModule.diffOp(data,d,pt)
    else:
        y = data

    model = ARIMA(y,order = (p,0,q))
    model_fit = model.fit()
    AIC_vals.append([d,model_fit.aic])

# find the optimal d (= dopt)value corresponding to minimum AIC and estimate parameters corresponding to the optimal d value  
AIC_vals = np.array(AIC_vals)
minind = np.nonzero(AIC_vals[:,1] == AIC_vals[:,1].min())
dopt = AIC_vals[minind[0],0]
y = FarimaModule.diffOp(data,dopt,pt)
model = ARIMA(y,order = (p,0,q))
model_fit = model.fit()

# extract parameters
arparams_est = model_fit.arparams
maparams_est = model_fit.maparams
sigma_est = model_fit.params[-1]
params = model_fit.params
theta = np.r_[arparams_est,maparams_est,dopt[0],sigma_est]

# extract confidence intervals
conf = model_fit.conf_int(0.05)
print(conf)

# Compute Hessian matrix of -2*loglikeli (Based on Whittle's approximate likelihood) and estimate the 95% confidence interval
one_sided = True
hess = FarimaModule.hessLw(p,q,theta,f,one_sided,pxx_den)
Sigma2 = np.linalg.inv(hess)
stds = np.diag(Sigma2)**0.5
conf_hessian = np.concatenate(((theta - 1.96*stds).reshape(1,-1).T, (theta + 1.96*stds).reshape(1,-1).T),axis = 1)

Lw = FarimaModule.objectiveFnc(theta,p,q,f,pxx_den,True,False)
print('Solution : ',theta)
print('Lw : ',Lw)