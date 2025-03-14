import pynof
import jax
import jax.numpy as jnp 
from jax import config
from scipy.differentiate import hessian,jacobian
from jax import jacfwd, jacrev
config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)
import numpy as np
from pynof.alex_pynof import alex_CJCKD5,alex_calce,alex_ocupacion_softmax,alex_der2_CJCKD5
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

dn2_d2gamma_ = lambda gamma : alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
# d2n_d2gamma = jax.hessian(dn2_d2gamma_,argnums=0)
# d2n_d2gamma = d2n_d2gamma(gamma)[0]
dn_dgamma_ = lambda gamma : alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
# dn_dgamma = jax.jacobian(dn_dgamma_,argnums=0)
# dn_dgamma = dn_dgamma(gamma)[0]

# calc_g = lambda gamma :calcocce(gamma,J_MO,K_MO,H_core,p)
# Hessian = opt.approx_fprime(gamma,calc_g,1e-6)
# print('Hessian',Hessian)
# assert 0 

def hessian_gamma(gamma, J_MO, K_MO, H_core, p):
    """
    Compute the Hessian of the energy with respect to gamma.
    """
    grad = jnp.zeros((p.nv))
    dn_dgamma = jax.jacfwd(dn_dgamma_,argnums=0)
    dn_dgamma,d2n_d2gamma  = dn_dgamma(gamma)
    # d2n_d2gamma = jnp.zeros_like(d2n_d2gamma)
    # dn2_d2gamma_ = lambda gamma : alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
    #                              p.nbf5, p.ndns, p.ncwo, p.HighSpin)
    # d2n_d2gamma = jax.hessian(dn2_d2gamma_,argnums=0)
    # d2n_d2gamma = d2n_d2gamma(gamma)[0]
    # # print(dn_dgamma(gamma))
    # print(d2n_d2gamma)
    # assert 0 
    n, _ = alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
    CJCK = lambda gamma : alex_der2_CJCKD5(gamma,n, p.ista, dn_dgamma, p.no1, p.ndoc, 
                                        p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo)
    
  
    # print(alex_der_CJCKD5_gamma(gamma,n, p.ista, dn_dgamma, p.no1, p.ndoc, 
    #                                     p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo))
    # hessian_D_gamma = hessian_D_gamma(gamma)
    # Dcj12r_dgamma,Dck12r_dgamma = hessian_D_gamma 
    # print('jax',j_Dcj12r_dgamma,j_Dck12r_dgamma)
    
    # print('-------------------------------------------------------------')
    Dcj12r_dgamma,Dck12r_dgamma = alex_der2_CJCKD5(n, dn_dgamma, d2n_d2gamma, p.no1, 
                                                   p.ndoc, p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo)
    
    # alex_der_CJCKD5_second_derivatives(n,dn_dgamma, d2n_d2gamma, 
    # p.no1, p.ndoc, p.nv, p.nbf5, p.ndns, p.ncwo)
    # print(Dcj12r_dgamma,Dck12r_dgamma)
    # assert 0 

    hessian = jnp.zeros((p.nv, p.nv))
    
    # Dcj12r_dgamma = Dcj12r_dgamma.at[jnp.diag_indices(p.nbf5)].set(0)
    # Dck12r_dgamma = Dck12r_dgamma.at[jnp.diag_indices(p.nbf5)].set(0)
    # # Contribution from dn_dgamma term
    hessian += jnp.einsum('ikl,i->kl', d2n_d2gamma, 2 * H_core + jnp.diagonal(J_MO), optimize=True)
    
    # # Contribution from Dcj12r term
    hessian += 2 * jnp.einsum('ijkl,ji->kl', Dcj12r_dgamma, J_MO, optimize=True)
    
    # # Contribution from Dck12r term
    hessian -= 2 * jnp.einsum('ijkl,ji->kl', Dck12r_dgamma, K_MO, optimize=True)
    
    return hessian

hessian_wrt_gamma = hessian_gamma(gamma, J_MO, K_MO, H_core, p)
print('another_hessian_gamma',hessian_wrt_gamma)

eigenvalues = jnp.linalg.eigvals(hessian_wrt_gamma)

# Print the eigenvalues
print("Eigenvalues:", eigenvalues)

# Check if all eigenvalues are non-negative
is_positive_semi_definite = jnp.all(eigenvalues >= 0)
print("Is the matrix positive semi-definite?", is_positive_semi_definite)