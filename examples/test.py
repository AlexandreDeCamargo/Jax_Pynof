
import pynof
import jax
import jax.numpy as jnp 
from jax import config
from scipy.differentiate import hessian,jacobian
from jax import jacfwd, jacrev
config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)
import numpy as np
from pynof.alex_pynof import alex_CJCKD5,alex_calce,test
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
p.set_ncwo(1)

p.RI = True



cj12,ck12,gamma,J_MO,K_MO,H_core,grad = pynof.compute_energy(mol,p,gradients=True)
n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)

def compute_hessians(n, cj12, ck12, J_MO, K_MO, H_core, p):
    # Define functions for each energy contribution
    def energy_total(n):
        return test(n, cj12, ck12, J_MO, K_MO, H_core, p)[0]
    
    def energy_cj12(n):
        return test(n, cj12, ck12, J_MO, K_MO, H_core, p)[1]
    
    def energy_ck12(n):
        return test(n, cj12, ck12, J_MO, K_MO, H_core, p)[2]
    
    # Compute Hessians
    hess_total = jax.hessian(energy_total)(n)
    hess_cj12 = jax.hessian(energy_cj12)(n)
    hess_ck12 = jax.hessian(energy_ck12)(n)
    
    return hess_total, hess_cj12, hess_ck12

print(compute_hessians(n, cj12, ck12, J_MO, K_MO, H_core, p)[0])

