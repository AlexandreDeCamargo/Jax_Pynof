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

def grad_n(n, J_MO, K_MO, H_core, p):
    # dDc_dn = lambda n : alex_der_CJCKD5(n, p.ista, dn_dgamma, p.no1, p.ndoc, 
    #                                     p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo,p)
    n = jnp.array(n)
    CJCK = lambda n :alex_CJCKD5(n, p.no1, p.ndoc, p.nsoc, 
                p.nbeta, p.nalpha, p.ndns, p.ncwo,p.MSpin)
    grad = jnp.zeros((p.nv))
    
    gradient_D_n = jax.jacobian(CJCK,argnums=0)
    
    Dcj12r_dn,Dck12r_dn = gradient_D_n(n)
    # jax.debug.print("🤯 dn_dn{x} 🤯", x=Dcj12r_dn)
    jax.debug.print("🤯 dck{y} 🤯", y=Dck12r_dn)
    # assert 0 
    grad += 2*H_core + jnp.diagonal(J_MO)
    
    # Dcj12r_dn = Dcj12r_dn.at[jnp.diag_indices(p.nbf5)].set(0)
    grad += 2 * jnp.einsum('ijk,jk->i', Dcj12r_dn, J_MO, optimize=True)
    
    # Dck12r_dn = Dck12r_dn.at[jnp.diag_indices(p.nbf5)].set(0)
    grad -= 2 * jnp.einsum('ijk,jk->i',Dck12r_dn, K_MO, optimize=True)

    return grad

# grad_wrt_n = grad_n(n, J_MO, K_MO, H_core, p)
# print('grad_wrt_n',grad_wrt_n)

def hessian_n(n, J_MO, K_MO, H_core, p):
    # dDc_dn = lambda n : alex_der_CJCKD5(n, p.ista, dn_dgamma, p.no1, p.ndoc, 
    #                                     p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo,p)

    n = jnp.array(n)
    CJK = lambda n :alex_CJCKD5(n, p.no1, p.ndoc, p.nsoc, 
                p.nbeta, p.nalpha, p.ndns, p.ncwo,p.MSpin)
    
    # print(CJK(n))
    cj12 = CJK(n)[0]
    ck12 = CJK(n)[1]
    
    hess = jnp.zeros((p.nv,p.nv))
    
    hess_cj12 = jax.hessian(cj12,argnums=0)
    hess_ck12 = jax.hessian(ck12,argnums=0)
    
    Dcj12r_dn,Dck12r_dn = hess_cj12(n), hess_ck12(n) 
    jax.debug.print("🤯 Dcj12r_dn{y} 🤯", y=Dcj12r_dn)
    # assert 0 
    # hess += 2*H_core + jnp.diagonal(J_MO)
    
    # Dcj12r_dn = Dcj12r_dn.at[jnp.diag_indices(p.nbf5)].set(0)
    hess += 2 * jnp.einsum('ijkl,jl->ik', Dcj12r_dn, J_MO, optimize=True)
    
    # Dck12r_dn = Dck12r_dn.at[jnp.diag_indices(p.nbf5)].set(0)
    hess -= 2 * jnp.einsum('ijkl,jl->ik',Dck12r_dn, K_MO, optimize=True)

    return hess
# d2ck[[[[ 2.00000000e+00  0.00000000e+00]
#    [ 0.00000000e+00  0.00000000e+00]]

#   [[-2.59165836e-02  2.46260955e+00]
#    [ 2.46260955e+00 -2.33998658e+02]]]


#  [[[-2.59165836e-02  2.46260955e+00]
#    [ 2.46260955e+00 -2.33998658e+02]]

#   [[ 0.00000000e+00  0.00000000e+00]
#    [ 0.00000000e+00 -1.42108547e-14]]]]
(hessian_n(n, J_MO, K_MO, H_core, p))
# assert 0 

# grad_E_n = lambda n : alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)
# dE_dn = opt.approx_fprime(n,grad_E_n,1e-5)
# print('Gradient of E wrt n',dE_dn)

hesss = lambda n: alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)
hess_ = jax.hessian(hesss,argnums=0)
H_n = hess_(n)
print('H_n',H_n)

assert 0
# eigenvalues = jnp.linalg.eigvals(H_n)

# # Print the eigenvalues
# print("Eigenvalues:", eigenvalues)

# # Check if all eigenvalues are non-negative
# is_positive_semi_definite = jnp.all(eigenvalues >= 0)
# print("Is the matrix positive semi-definite?", is_positive_semi_definite)
# print(hess_(n))
# H_E_n = lambda n : grad_n(n, J_MO, K_MO, H_core, p)
# H_dn = opt.approx_fprime(n,H_E_n,1e-10)
# print('Hessian of E wrt n',H_dn)
# grad_wrt_n [-6.86619997 -1.71001114]

# def cj12_term(n, J_MO, p):
#     cj12, _ = alex_CJCKD5(n, p.no1, p.ndoc, p.nsoc, 
#                           p.nbeta, p.nalpha, p.ndns, p.ncwo, p.MSpin)
#     cj12 = cj12.at[jnp.diag_indices_from(cj12)].set(0)
#     return jnp.einsum('ij,ji->', cj12, J_MO, optimize=True)

# def ck12_term(n, K_MO, p):
#     _, ck12 = alex_CJCKD5(n, p.no1, p.ndoc, p.nsoc, 
#                           p.nbeta, p.nalpha, p.ndns, p.ncwo, p.MSpin)
#     ck12 = ck12.at[jnp.diag_indices_from(ck12)].set(0)
#     return jnp.einsum('ij,ji->', ck12, K_MO, optimize=True)

# hess_cj12_fn = jax.hessian(cj12_term, argnums=0)
# hess_ck12_fn = jax.hessian(ck12_term, argnums=0)

# H_cj12 = hess_cj12_fn(n, J_MO, p)
# H_ck12 = hess_ck12_fn(n, K_MO, p)

# print("Hessian of cj12 term:", H_cj12)
# print("Hessian of ck12 term:", H_ck12)