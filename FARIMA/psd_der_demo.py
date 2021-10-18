"""
This script test the derivatives (Jacobian and Hessian) of FARIMA power spectral density w.r.t. its parameters

Author: Abhinav Gupta (Created: 14 Oct 2021)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import optimize
from scipy.optimize.nonlin import Jacobian
import  statsmodels.api as sm
import FarimaModule
from  statsmodels.tsa.arima.model import ARIMA
import numdifftools as nd

np.random.seed(1)

p = 3                               # autoregressive order
q = 3                               # moving-average order
d = 0.3
arparams = np.array([0.30,0.10,0.40])    # autoregressive coefficients
maparams = np.array([0.99,0.2,0.9])      # moving-average coefficients
ar = np.r_[1, -arparams]
ma = np.r_[1, maparams]
sigma_eps = 1.0                     # varaince of white noise
N = 1000                            # number of time-steps at which data will be generated
x = np.arange(1,N+1,1)

# generate data
data = sm.tsa.arma_generate_sample(ar,ma,N,sigma_eps)
plt.plot(x,data)
plt.show()

if (d != 0):
    pt = 100
    data = FarimaModule.invdiffOp(data,d,pt)
plt.plot(x,data)
plt.show()

# compute periodogram
fs = 1
f, I = FarimaModule.periodogramAG(data,fs,True)
f = f[1:]  # remove the zero
I = I[1:]  # remove the value at zero frequency

# power spectral density at different frequencies
one_sided  = True
pxx_den_theory = FarimaModule.pxx_denFARIMA(p,d,q,ar,ma,f,sigma_eps,one_sided)

# define iota and angular frequencies
iota = (-1)**0.5
omega = f*2*np.pi
exp_neg_iota_omega = np.exp(-iota*omega)

# derivative w.r.t. d
term1 = 1 - I/pxx_den_theory
term2 = np.log(abs(1-exp_neg_iota_omega))
partial_Lw_d = -2*np.sum(term1*term2)

# derivative w.r.t. shi_k for k = 1 to q 
krange = range(1,q+1)
ma_pol = FarimaModule.movingAvgPol(q,ma,exp_neg_iota_omega)
mterm2 = abs(ma_pol)**(-2)
partial_Lw_shi = []
for k in krange:
    mterm3 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + (np.conj(exp_neg_iota_omega)**k)*ma_pol
    partial_Lw_shi_k = np.sum(term1*mterm2*mterm3)
    partial_Lw_shi.append(partial_Lw_shi_k)
partial_Lw_shi = np.array(partial_Lw_shi)

# derivative w.r.t. phi_k for k = 1 to p
krange = range(1,p+1)
ar_pol = FarimaModule.autoregressPol(p,ar,exp_neg_iota_omega)
aterm2 = abs(ar_pol)**(-2)
partial_Lw_phi = []
for k in krange:
    aterm3 = (exp_neg_iota_omega**k)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**k)*ar_pol
    partial_Lw_phi_k = np.sum(term1*aterm2*aterm3)
    partial_Lw_phi.append(partial_Lw_phi_k)
partial_Lw_phi = np.array(partial_Lw_phi)

# parital derivative of Lw w.r.t. sigma_eps
partial_Lw_sigma_eps = 2*np.sum(term1)/sigma_eps

# arrange derivatives in a list
jacobian = partial_Lw_phi.tolist() + partial_Lw_shi.tolist() + [partial_Lw_d] +  [partial_Lw_sigma_eps] 

#########################################################################################################

# Hessian of the Whittle's approximate likelihood
# second partial w.r.t d
term1 = I/pxx_den_theory
term2 = np.log(abs(1 - exp_neg_iota_omega))
partial2_Lw_d2 = 4*(np.sum(term1*(term2**2)))

# mixed second partial of d and shi_k(s)
krange = range(1,q+1)
ma_pol = FarimaModule.movingAvgPol(q,ma,exp_neg_iota_omega)
mdterm3 = abs(ma_pol)**(-2)
partial2_Lw_shi_d = []
for k in krange:
    mdterm4 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**k)*ma_pol
    partial2_Lw_shik_d = -2*np.sum(term1*term2*mdterm3*mdterm4)
    partial2_Lw_shi_d.append(partial2_Lw_shik_d)

# mixed second partial of d and phi_k(s)
krange = range(1,p+1)
ar_pol = FarimaModule.autoregressPol(p,ar,exp_neg_iota_omega)
adterm3 = abs(ar_pol)**(-2)
partial2_Lw_phi_d = []
for k in krange:
    adterm4 = (exp_neg_iota_omega**k)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**k)*ar_pol
    partial2_Lw_phik_d = -2*np.sum(term1*term2*adterm3*adterm4)
    partial2_Lw_phi_d.append(partial2_Lw_phik_d)

# mixed second partial of d and sigma_eps
partial2_Lw_sigmaeps_d = -4*np.sum(term1*term2/sigma_eps)

# second partial derivatives of shi_k(s) and shi_l(s)
krange = range(1,q+1)
lrange = krange

partial2_Lw_shi_shi = np.zeros((q,q))
for k in krange:
    for l in lrange:
        mmterm1 = -1 + 2*term1
        mmterm2 = (abs(ma_pol))**(-4)
        mmterm3 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**k)*ma_pol
        mmterm4 = (exp_neg_iota_omega**l)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**l)*ma_pol
        mmterm5 = 1 - term1
        mmterm6 = (abs(ma_pol))**(-2)
        mmterm7 = exp_neg_iota_omega**(l-k) + np.conj(exp_neg_iota_omega**(l-k))
        partial2_shik_shil = np.sum(mmterm1*mmterm2*mmterm3*mmterm4) + np.sum(mmterm5*mmterm6*mmterm7)
        partial2_Lw_shi_shi[k-1,l-1] = partial2_shik_shil

# mixed second partial derivatives of phi_k(s) and shi_l(s) 
krange = range(1,q+1)   # MA coefficients 
lrange = range(1,p+1)   # AR coefficients
partial2_Lw_shi_phi = np.zeros((q,p))
for k in krange:
    for l in lrange:
        amterm2 = (abs(ma_pol))**(-2)
        amterm3 = (abs(ar_pol))**(-2)
        amterm4 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**k)*ma_pol  # MA term
        amterm5 = (exp_neg_iota_omega**l)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**l)*ar_pol  # AR term
        partial2_Lw_phil_shik = np.sum(term1*amterm2*amterm3*amterm4*amterm5)
        partial2_Lw_shi_phi[k-1,l-1] = partial2_Lw_phil_shik


# mixed second partial derivatives of shi_k(s) and sigma_epsilon
k = 1
smterm2 = 1/sigma_eps
smterm3 = abs(ma_pol)**(-2)
partial2_Lw_sigmaeps_shi = []
for k in krange:
    smterm4 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**k)*ma_pol
    partial2_sigmaeps_shik = 2*np.sum(term1*smterm2*smterm3*smterm4)
    partial2_Lw_sigmaeps_shi.append(partial2_sigmaeps_shik)

# mixed second partial derivatives of phi_k(s) and phi_l(s)
krange = range(1,p+1)
lrange = krange
aaterm1 = abs(ar_pol)**(-4)
aaterm4 = 1 - term1
aaterm5 = abs(ar_pol)**(-2)
partial2_Lw_phi_phi = np.zeros((p,p))
for k in krange:
    for l in lrange:
        
        aaterm2 = (exp_neg_iota_omega**k)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**k)*ar_pol
        aaterm3 = (exp_neg_iota_omega**l)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**l)*ar_pol
        aaterm6 = exp_neg_iota_omega**(l-k) + np.conj(exp_neg_iota_omega**(l-k))
        partial2_Lw_phik_phil = np.sum(aaterm1*aaterm2*aaterm3) - np.sum(aaterm4*aaterm5*aaterm6)
        partial2_Lw_phi_phi[k-1,l-1] = partial2_Lw_phik_phil


# mixed second partial derivatives of LW w.r.t. phi_k(s) and sigma_eps
krange = range(1,p+1)
saterm2 = 1/sigma_eps
saterm3 = abs(ar_pol)**(-2)
partial2_Lw_sigmaeps_phi = np.zeros((p,1))
for k in krange:
    saterm4 = (exp_neg_iota_omega**k)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**k)*ar_pol
    partial2_Lw_sigmaeps_phi[k-1] = 2*np.sum(term1*saterm2*saterm3*saterm4)

# second partial derivative of Lw w.r.t. sigma_eps
partial2_Lw_sigmaeps2 = -2*len(f)/sigma_eps**2 + (6/sigma_eps**2)*np.sum(term1)

# arrange all the elemenst into a  matrix - Hessian matrix
partial2_Lw_shi_phi = np.array(partial2_Lw_shi_phi)
partial2_Lw_phi_phi = np.array(partial2_Lw_phi_phi)
partial2_Lw_shi_shi = np.array(partial2_Lw_shi_shi)
partial2_Lw_phi_d = np.array(partial2_Lw_phi_d).reshape(1,-1)
partial2_Lw_shi_d = np.array(partial2_Lw_shi_d).reshape(1,-1)
partial2_Lw_sigmaeps_phi = np.array(partial2_Lw_sigmaeps_phi).reshape(1,-1)
partial2_Lw_sigmaeps_shi = np.array(partial2_Lw_sigmaeps_shi).reshape(1,-1)
partial2_Lw_d2 = np.array(partial2_Lw_d2).reshape(1,-1) 
partial2_Lw_sigmaeps_d = np.array(partial2_Lw_sigmaeps_d).reshape(1,-1)
partial2_Lw_sigmaeps2 = np.array(partial2_Lw_sigmaeps2).reshape(1,-1)

Hessian_mat1 = np.concatenate((partial2_Lw_phi_phi,partial2_Lw_shi_phi,partial2_Lw_phi_d,partial2_Lw_sigmaeps_phi))
Hessian_mat2 = np.concatenate((partial2_Lw_shi_phi.T,partial2_Lw_shi_shi,partial2_Lw_shi_d,partial2_Lw_sigmaeps_shi))
Hessian_mat3 = np.concatenate((partial2_Lw_phi_d.T,partial2_Lw_shi_d.T,partial2_Lw_d2,partial2_Lw_sigmaeps_d))
Hessian_mat4 = np.concatenate((partial2_Lw_sigmaeps_phi.T,partial2_Lw_sigmaeps_shi.T,partial2_Lw_sigmaeps_d,partial2_Lw_sigmaeps2))
Hessian_mat = np.concatenate((Hessian_mat1,Hessian_mat2,Hessian_mat3,Hessian_mat4),axis = 1)

# compare FarimaModule definition of Hessian with the one calculated above
theta = np.r_[arparams,maparams,d,sigma_eps] 
diff = FarimaModule.hessLw(p,q,theta,f,True,I) - Hessian_mat
a = np.isclose(diff,np.zeros((8,8)))
print(a)

# compute eigenvalues of Hessian
w, v = np.linalg.eig(Hessian_mat)
print(w)

# compute inverse of the Hessian matrix
Sigma = np.linalg.inv(Hessian_mat)
print(Sigma)

# compute jacobian and Hessian using numerical scheme - numdifftools 
def func(x):
    one_sided = True
    return  FarimaModule.objectiveFnc(x,p,q,f,I,one_sided,False)

theta = arparams.tolist() + maparams.tolist() + [d] + [sigma_eps]
jac = nd.Jacobian(func)(theta)
hess = nd.Hessian(func)(theta)

diff = hess - Hessian_mat
print(diff)
a = np.isclose(diff,np.zeros((8,8)))
print(a)