import pynof
import jax
import jax.numpy as jnp 
from jax import config
# from scipy.differentiate import hessian,jacobian
from jax import jacfwd, jacrev
config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)
import numpy as np
from pynof.alex_pynof import alex_CJCKD5,alex_calce,test,alex_calcoccg,alex_calcocce
from pynof.pnof import PNOFi_selector,ocupacion,der_PNOFi_selector,calce,calcoccg,calcocce

from scipy.optimize import minimize 
from numdifftools import Jacobian, Hessian

import scipy.optimize as opt

# config.DISABLE_JIT = True

mol = pynof.molecule("""
0 1
  H  0.0000 	0.0000 	0.0000
  H  0.0000 	0.0000 	0.7000
""")


p = pynof.param(mol,"cc-pvdz")

p.ipnof = 5
p.set_ncwo(2)

p.RI = True

cj12,ck12,gamma,J_MO,K_MO,H_core,grad = pynof.compute_energy(mol,p,gradients=True)
n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)

#Define energy as a lambda function
E_n = lambda n: alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

#First calculating the gradient using jacfwd and then taking the second derivative to find the Hessian
grad_E_n = jax.jacfwd(E_n,argnums=0)
print(grad_E_n(n))
hessian = jax.jacrev(grad_E_n,argnums=0)
print('Hessian_using_gradient',hessian(n)) 

#Calculating the Hessian using the jax.hessian function
hess_ = jax.hessian(E_n,argnums=0)
H_n = hess_(n)
print('H_n',H_n)


#Finite Differences Method 
E_n = lambda n : alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)
dE_dn = opt.approx_fprime(n,E_n,1e-5)
print('Gradient of E wrt n',dE_dn)

#Example using the gradient calculate in pynof for gamma
d_E_gamma = lambda gamma :alex_calcoccg(gamma, J_MO, K_MO, H_core, p)
H_gamma =jax.jacobian(d_E_gamma)(gamma)
print('H_gamma',H_gamma)