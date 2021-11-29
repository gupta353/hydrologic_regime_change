"""
This script computes periodogram of given data, tests basinhopping algorithm for optimization of the data
Whittle's approximate likelihood function to find FARIMA model parameters, and test the optimization using the
derivative method 

Author: Abhinav Gupta (Created: 5 Oct 2021)

Note: periodogram calculation by scipy do not use 1/2/pi factor
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

""" # first order autoregressive data
pxx_den = []
a = 0.6 # value of the coefficient
N = 1000 # number of time-steps
mu = 0    # mean of the white noisea
sigma = 10 # standard deviation of white noise
sigx2 = sigma**2/(1 - a**2)
x = np.arange(0,N+1)

for iter in range(0,1000):

    data = [0]
    for ind in range(0,N):
        data.append(a*data[ind] + np.random.normal(loc = 0, scale = sigma))
    data = np.array(data)

    # sinusoidal data
  
    x = np.arange(0,10,0.1)
    # data = np.sin(x) + np.sin(2*np.pi*x) + np.sin(4*np.pi*x) + np.sin(np.pi*x/4)
    data = np.sin(4*np.pi*x)

    
    plt.plot(x,data)
    plt.show()
    

    # compute periodogram
    delta_t = x[1] - x[0]
    fs = 1/delta_t
    f, pxx_den_tmp = signal.periodogram(data,fs)
    pxx_den.append(pxx_den_tmp)

pxx_den = np.array(pxx_den)
pxx_den = pxx_den.mean(axis = 0)
plt.plot(f*2*np.pi,pxx_den)


# theoretical values
pxx_den_theory = 2*sigx2*(1 - a**2)/(1 - 2*a*np.cos(2*np.pi*f) + a**2)
plt.plot(f*2*np.pi,pxx_den_theory)
plt.show()
"""
# generation of AR(3) process data
"""
a1 = 0.25
a2 = 0.20
a3 = 0.40
sigma = 1
mu = 0
N= 1000
data = [0,0,0]
for ind in range(2,N+2):
    data.append(a1*data[ind-1] + a2*data[ind-2] + a3*data[ind-3] + np.random.normal(mu,sigma))

plt.plot(data)
plt.show()
"""

# FARIMA model

p = 4                               # autoregressive order
q = 4                              # moving-average order
d = 0.11
arparams = np.array([3.0096,-3.76,2.39,-0.64])    # autoregressive coefficients
maparams = np.array([-1.47,0.89,-0.069,-0.1151])        # moving-average coefficients
ar = np.r_[1, -arparams]
ma = np.r_[1, maparams]
sigma = 3.83                         # varaince of white noise
N = 3650                            # number of time-steps at which data will be generated
x = np.arange(1,N+1,1)

pxx_den = []
for iter in range(0,100):
    data = sm.tsa.arma_generate_sample(ar,ma,N,sigma)
    #plt.plot(x,data)

    # apply inverse of the differencing operator
    if (d != 0):
        pt = 100
        data = FarimaModule.invdiffOp(data,d,pt)
    #plt.plot(x,data)
    #plt.show()

    # compute periodogram
    fs = 1
    f,pxx_den_tmp = signal.periodogram(data,fs)
    pxx_den.append(pxx_den_tmp)

pxx_den = np.array(pxx_den)
pxx_den = pxx_den.mean(axis = 0)

plt.loglog(f[1:],pxx_den[1:])
plt.show()

# theoretical power spectral densityof FARIMA model 
pxx_epsilon = sigma**2
omegas = f*2*np.pi
iota = (-1)**0.5
exp_neg_iota_omega = np.exp(-iota*omegas)

# auto-regressive polynomial
pow = np.arange(0,p+1)    # power of exponential
phi_p = []
for ind in range(0,exp_neg_iota_omega.shape[0]):
    vals = exp_neg_iota_omega[ind]**pow
    vals = ar*vals 
    phi_p.append(np.sum(vals))
phi_p = np.array(phi_p)

# moving average polynomials
pow = np.arange(0,q+1)    # power of exponential
shi_q = []
for ind in range(0,exp_neg_iota_omega.shape[0]):
    vals = exp_neg_iota_omega[ind]**pow
    vals = ma*vals 
    shi_q.append(np.sum(vals))
shi_q = np.array(shi_q)

pxx_den_theory = 2*(abs(1 - exp_neg_iota_omega))**(-2*d)*(abs(shi_q))**2*(abs(phi_p))**(-2)*pxx_epsilon
#pxx_den_theory = (abs(1 - exp_neg_iota_omega))**(-2*d)*pxx_epsilon

plt.loglog(f[1:],pxx_den_theory[1:])
plt.title('Theoretical FARIAM spectra')
plt.show()

plt.scatter(pxx_den[1:],pxx_den_theory[1:])
plt.show()


# Compute Whittle's approximate likelihood for different parameter values
""" paramvals = np.arange(0,0.5,0.01)
Lw = []
for ind in range(0,paramvals.shape[0]):
    paramtmp = paramvals[ind]
    d = paramtmp
    theta = np.r_[arparams,maparams,d,sigma]
    Lw.append(func(theta))
plt.plot(paramvals,Lw)
plt.show() """

""" paramvals1 = np.arange(-1,1,0.05)
paramvals2 = np.arange(0,1,0.05)
Lw = []
for ind1 in range(0,paramvals1.shape[0]):
    for ind2 in range(0,paramvals2.shape[0]):
        arparams[0] = paramvals1[ind1]
        maparams[1] = paramvals2[ind2]
        theta = np.r_[arparams,maparams,d,sigma]
        Lw.append([paramvals1[ind1],paramvals2[ind2],func(theta)])
Lw = np.array(Lw)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Lw[:,0],Lw[:,1],Lw[:,2])
min_ind = np.nonzero(Lw[:,2]==Lw[:,2].min())
ax.scatter(Lw[min_ind,0],Lw[min_ind,1],Lw[min_ind,2])

ax.set_xlabel('ar param')
ax.set_ylabel('ma param')
ax.set_zlabel('Lw')
plt.show()
 """

# compute derivatives of Whittle's approximate likeihood w.r.t. different parameters
""" paramvals = np.arange(0,0.5,0.01)
der_Lw = []
for ind in range(0,paramvals.shape[0]):
    paramtmp = paramvals[ind]
    d = paramtmp
    theta = np.r_[arparams,maparams,d,sigma]
    der_Lw.append(funcder(theta))
der_Lw = np.array(der_Lw)
plt.plot(paramvals,der_Lw[:,6])
plt.plot(paramvals,[0]*paramvals.shape[0])
plt.show() """

""" theta = np.concatenate((arparams,maparams,[d]))
def funcder(x,p,q,f,sigma_eps,one_sided,I):
    return derLw(p,q,x,f,sigma_eps,one_sided,I) """

""" args = (p,q,f,sigma_eps,True,I)
conv_test = False
theta = [0.5,0.5,0.5,0.5,0.5,0.5,0.1]
while (conv_test == False):
    roots = theta + np.random.normal(0,0.1,(7,))
    roots, info, ier, mesg = optimize.fsolve(funcder, roots, args, full_output=True, maxfev = 10000 ,diag = [0.08]*7)
    val = funcder(roots,p,q,f,sigma_eps,True,I)
    conv_test = np.isclose(val,[0]*7)
    conv_test = all(conv_test)
    print(roots, val) """