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
# print('cj12',cj12)
# print('ck12',ck12)
# assert 0 
n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
# print('n',n.shape)
# assert 0 
ck = lambda n:  alex_CJCKD5(n, p.no1, p.ndoc, p.nsoc, 
                      p.nbeta,p.nalpha, p.ndns, p.ncwo, p.MSpin)


print(jax.make_jaxpr(jax.jacfwd(ck))(n))
assert 0 
# D_ck_ = jax.jacrev(ck)
# print(D_ck_)
# assert 0 
# D2_2ck = jax.jacfwd(D_ck_)(n)
D2_2ck = jax.hessian(ck)(n)
print(D2_2ck)
# print(D_ck_)
assert 0 

# [[0.         0.        ]
#  [0.05129322 4.87275631]
#  [0.05129322 4.87275631]
#  [0.         0.        ]]
# assert 0 
# D_ck = opt.approx_fprime(n,ck,1e-5)
# print(D_ck)
# assert 0 
# n = n + 1e-3
# gamma = gamma + 1e-2

# grad_gamma = lambda gamma: alex_calcocce(gamma,cj12, ck12,J_MO,K_MO,H_core,p)
# grads_gamma = jax.grad(grad_gamma,argnums=0)
# print(grads_gamma(gamma))
# assert 0 
# hess_gamma = lambda gamma: alex_calcoccg(gamma, J_MO, K_MO, H_core, p)
# hess_g = jax.jacobian(hess_gamma,argnums=0)
# H_g = hess_g(gamma)
# print('H_g',H_g)

# is_symmetric = np.allclose(H_g, H_g.T)
# print("Is the matrix symmetric?", is_symmetric)

# if is_symmetric: 

#     eigenvalues = jnp.linalg.eigvals(H_g)

#     # Print the eigenvalues
#     print("Eigenvalues:", eigenvalues)

#     # Check if all eigenvalues are non-negative
#     is_positive_semi_definite = jnp.all(eigenvalues >= 0)
#     print("Is the matrix positive semi-definite?", is_positive_semi_definite)


# assert 0 
# print(n)
# assert 0 
# n = n + 1e-3
E_n = lambda n: alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)
grad_E_n = jax.jacfwd(E_n,argnums=0)
print(grad_E_n(n))
essian = jax.jacrev(grad_E_n,argnums=0)
print(essian(n)) 
hess_ = jax.hessian(E_n,argnums=0)
H_n = hess_(n)
print('H_n',H_n)
is_symmetric = np.allclose(H_n, H_n.T)
print("Is the matrix symmetric?", is_symmetric)

if is_symmetric: 

    eigenvalues = jnp.linalg.eigvals(H_n)

    # Print the eigenvalues
    print("Eigenvalues:", eigenvalues)

    # Check if all eigenvalues are non-negative
    is_positive_semi_definite = jnp.all(eigenvalues >= 0)
    print("Is the matrix positive semi-definite?", is_positive_semi_definite)

# assert 0 
E_n = lambda n : alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)
dE_dn = opt.approx_fprime(n,E_n,1e-5)
print('Gradient of E wrt n',dE_dn)
# minimize(fun, x0, args=(), method=None, jac=None, hess=None,
#              hessp=None, bounds=None, constraints=(), tol=None,
#              callback=None, options=None)
def E_n_(n):
    return alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

# Now you can use E_n in the minimize function
# Hess = minimize(E_n_, n, method='trust-exact', tol=1e-5)
# fmin_ncg(f, x0, fprime, fhess_p=None, fhess=None, args=(), avextol=1e-5,
#              epsilon=_epsilon, maxiter=None, full_output=0, disp=1, retall=0,
#              callback=None, c1=1e-4, c2=0.9):
# Hess = opt.fmin_ncg(E_n,n,H_n[1])
# Hess = minimize(E_n,n, tol=1e-5)
# Hess = minimize(E_n,n,tol=1e-5)
# print(Hess)
assert 0 
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
    
    return hess_total

# print(compute_hessians(n, cj12, ck12, J_MO, K_MO, H_core, p))
is_symmetric = np.allclose(H_n, H_n.T)
print("Is the matrix symmetric?", is_symmetric)

if is_symmetric: 

    eigenvalues = jnp.linalg.eigvals(H_n)

    # Print the eigenvalues
    print("Eigenvalues:", eigenvalues)

    # Check if all eigenvalues are non-negative
    is_positive_semi_definite = jnp.all(eigenvalues >= 0)
    print("Is the matrix positive semi-definite?", is_positive_semi_definite)

