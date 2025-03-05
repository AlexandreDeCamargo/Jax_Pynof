import pynof
import jax
import jax.numpy as jnp 
from jax import config
from scipy.differentiate import hessian,jacobian
from jax import jacfwd, jacrev
config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)
import numpy as np
from pynof.alex_pynof import Alex_ocupacion_softmax,compute_hessian_finite_diff,alex_calchess_finite_diff,compute_gradient_finite_diff,alex_calcoccg_finite_diff,alex_der2_CJCKD5,alex_CJCKD5,alex_JKH_MO_RI,alex_ocupacion_softmax,alex_der_CJCKD5_second_derivatives,alex_der_CJCKD5,alex_der_CJCKD5_gamma
from pynof.pnof import PNOFi_selector,ocupacion,der_PNOFi_selector,calce,calcoccg,calcocce

from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian

import scipy.optimize as opt

# config.DISABLE_JIT = True

mol = pynof.molecule("""
0 1
  H  0.0000 	0.0000 	0.0000
  H  0.0000 	0.0000 	0.5000
""")


p = pynof.param(mol,"cc-pvdz")

p.ipnof = 5
p.set_ncwo(1)

p.RI = True



cj12,ck12,gamma,J_MO,K_MO,H_core,grad = pynof.compute_energy(mol,p,gradients=True)



dn2_d2gamma_ = lambda gamma : alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
# d2n_d2gamma = jax.hessian(dn2_d2gamma_,argnums=0)
# d2n_d2gamma = d2n_d2gamma(gamma)[0]
dn_dgamma_ = lambda gamma : alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
# dn_dgamma = jax.jacobian(dn_dgamma_,argnums=0)
# dn_dgamma = dn_dgamma(gamma)[0]
# print(dn_dgamma)
# assert 0 
# n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
# n = jnp.array(n)
# dn_dgamma = jnp.array(dn_dgamma)
# print('jax',alex_der_CJCKD5(n, p.ista, dn_dgamma, p.no1, p.ndoc, p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo,p))

# gradient = compute_gradient_finite_diff(gamma, J_MO, K_MO, H_core, p)
# print(gradient)
# def calcocce(gamma,J_MO,K_MO,H_core,p):
#     # n,_ = Alex_ocupacion_softmax(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
#     n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
#     # n = np.array([0.98958557,0.01041443])
#     cj12,ck12 = PNOFi_selector(n,p)
#     E = calce(n,cj12,ck12,J_MO,K_MO,H_core,p)
#     # print('EEE',E)
#     return E
# print('Pnof',calcoccg(gamma,J_MO,K_MO,H_core,p))

calc_g = lambda gamma :calcoccg(gamma,J_MO,K_MO,H_core,p)
Hessian = opt.approx_fprime(gamma,calc_g,1e-6)
print('Hessian',Hessian)
assert 0 

# calc_E = lambda gamma :calcocce(gamma,J_MO,K_MO,H_core,p)
# opt = minimize(calc_E, x0=gamma, method='L-BFGS-B')
# print(opt)
# B = opt.hess_inv.todense()  # Convert the linear operator to a dense matrix

# Print the inverse Hessian
# print("Inverse Hessian (B):")
# print(B)
# assert 0 

# print(alex_calcoccg_finite_diff(gamma, J_MO, K_MO, H_core, p,))
# assert 0 

# assert 0 
# print(dn_dgamma)
# assert 0 
n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)

# print('pnof',der_PNOFi_selector(n,dn_dgamma,p))

# assert 0 
def grad_gamma(gamma, J_MO, K_MO, H_core, p):
    """
    Compute the Hessian of the energy with respect to gamma.
    """
    grad = jnp.zeros((p.nv))
    dn_dgamma = jax.jacobian(dn_dgamma_,argnums=0)
    dn_dgamma = dn_dgamma(gamma)[0]
    
    # dn_dgamma = jnp.array(dndg)

    # jax.debug.print("🤯 dn_dgamma{y} 🤯", y=dn_dgamma)
    # jax.debug.print("🤯 n{x} 🤯", x=n)
    # print('gamma',gamma)
    Dcj12r, Dck12r = alex_der_CJCKD5(n, p.ista, dn_dgamma, p.no1, p.ndoc, 
                                     p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo,p)
    Dcj12r, Dck12r = jnp.asarray(Dcj12r), jnp.asarray(Dck12r)
    
    grad += jnp.einsum('ik,i->k',dn_dgamma,2*H_core+jnp.diagonal(J_MO),optimize=True)

    # 2 dCJ_dgamma J_MO

    Dcj12r = Dcj12r.at[jnp.diag_indices(p.nbf5)].set(0)
    # Dcj12r = Dcj12r.at[jnp.diag_indices(p.nbf5)].set(0)
    grad += 2 * jnp.einsum('ijk,ji->k', Dcj12r, J_MO, optimize=True)
    
    Dck12r = Dck12r.at[jnp.diag_indices(p.nbf5)].set(0)
    # Dck12r = Dck12r.at[jnp.diag_indices(p.nbf5)].set(0)
    grad -= 2 * jnp.einsum('ijk,ji->k', Dck12r, K_MO, optimize=True)
    
    return grad

grad_wrt_gamma = grad_gamma(gamma, J_MO, K_MO, H_core, p)
# print('grad_wrt_gamma',grad_wrt_gamma)
# assert 0 

# 
# hessian_E_gamma = jax.jacobian(grad_gamma, argnums=0)
# H_gamma = hessian_E_gamma(n, J_MO, K_MO, H_core, p)
# print("Hessian_gamma:", H_gamma)
# eigenvalues = jnp.linalg.eigvals(H_gamma)

# # # Print the eigenvalues
# print("Eigenvalues:", eigenvalues)

# # # Check if all eigenvalues are non-negative
# is_positive_semi_definite = jnp.all(eigenvalues >= 0)
# print("Is the matrix positive semi-definite?", is_positive_semi_definite)
# assert 0 


def grad_n(n, J_MO, K_MO, H_core, p):
    # dDc_dn = lambda n : alex_der_CJCKD5(n, p.ista, dn_dgamma, p.no1, p.ndoc, 
    #                                     p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo,p)
    CJCK = lambda n :alex_CJCKD5(n, p.no1, p.ndoc, p.nsoc, 
                p.nbeta, p.nalpha, p.ndns, p.ncwo,p.MSpin)
    grad = jnp.zeros((p.nv))
    
    gradient_D_n = jax.jacobian(CJCK,argnums=0)
    Dcj12r_dn,Dck12r_dn = gradient_D_n(n)

    # print(gradient_D_n)
    # assert 0 
    # Dcj12r_dn,Dck12r_dn = gradient_D_n
    # print(Dcj12r_dn)
    # print(Dck12r_dn)
    # assert 0 
    
    grad += 2*H_core + jnp.diagonal(J_MO)
    
    Dcj12r_dn = Dcj12r_dn.at[jnp.diag_indices(p.nbf5)].set(0)
    grad += 2 * jnp.einsum('ijk,jk->i', Dcj12r_dn, J_MO, optimize=True)
    
    Dck12r_dn = Dck12r_dn.at[jnp.diag_indices(p.nbf5)].set(0)
    grad -= 2 * jnp.einsum('ijk,jk->i',Dck12r_dn, K_MO, optimize=True)

    return grad

grad_wrt_n = grad_n(n, J_MO, K_MO, H_core, p)
print('grad_wrt_n',grad_wrt_n)

hessian_E_n = jax.jacfwd(grad_n, argnums=0)
H_n = hessian_E_n(n, J_MO, K_MO, H_core, p)

print("Hessian_n:", H_n)
# assert 0

def hessian_n(n, J_MO, K_MO, H_core, p):
    # CJCK= lambda n : alex_der_CJCKD5(n, p.ista, dn_dgamma, p.no1, p.ndoc, 
    #                                     p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo,p)
    
    CJCK = lambda n :alex_CJCKD5(n, p.no1, p.ndoc, p.nsoc, 
                p.nbeta, p.nalpha, p.ndns, p.ncwo,p.MSpin)
    hessian = jnp.zeros((p.nv, p.nv))
    hessian_D_n = jax.hessian(CJCK,argnums=0)
    hessian_D_n = hessian_D_n(n)
    Dcj12r_dn,Dck12r_dn = hessian_D_n

    # print(Dcj12r_dn.shape)
    # assert 0 
    # print(Dcj12r_dn)
    # print(Dck12r_dn)
    # assert 0 
    
    # hessian += 2*H_core + jnp.diagonal(J_MO)
    
    # Dcj12r_dn = Dcj12r_dn.at[jnp.diag_indices(p.nbf5)].set(0)
    # Dck12r_dn = Dck12r_dn.at[jnp.diag_indices(p.nbf5)].set(0)
    
    hessian += 2 * jnp.einsum('ijkl,jl->ik', Dcj12r_dn, J_MO, optimize=True)
    hessian -= 2 * jnp.einsum('ijkl,jl->ik', Dck12r_dn, K_MO, optimize=True)
    # Add the contributions from Dcj12r_dn and Dck12r_dn
    # hessian += 2 * jnp.einsum('abcde,de->a', Dcj12r_dn, J_MO, optimize=True)
    # hessian -= 2 * jnp.einsum('abcde,de->a', Dck12r_dn, K_MO, optimize=True)
    

    return hessian

hessian_wrt_n = hessian_n(n, J_MO, K_MO, H_core, p)
# print('another_hessian_n',hessian_wrt_n)
# eigenvalues = jnp.linalg.eigvals(hessian_wrt_n)

# # Print the eigenvalues
# print("Eigenvalues:", eigenvalues)

# # # Check if all eigenvalues are non-negative
# is_positive_semi_definite = jnp.all(eigenvalues >= 0)
# print("Is the matrix positive semi-definite?", is_positive_semi_definite)
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

# eigenvalues = jnp.linalg.eigvals(hessian_wrt_gamma)

# # Print the eigenvalues
# print("Eigenvalues:", eigenvalues)

# # Check if all eigenvalues are non-negative
# is_positive_semi_definite = jnp.all(eigenvalues >= 0)
# print("Is the matrix positive semi-definite?", is_positive_semi_definite)

# [[ 0.01638187 -0.01638187]
#  [-0.01638187  0.01638187]]