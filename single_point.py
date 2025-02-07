import pynof
import jax
import jax.numpy as jnp 
from jax import config
config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)

from pynof.alex_pynof import calcocce,calce,calcoccg,calcoccg_jax,jax_ocupacion_softmax
from pynof.pnof import PNOFi_selector, ocupacion

# config.DISABLE_JIT = True

mol = pynof.molecule("""
0 1
  H  0.0000 	0.0000 	0.3740
  H  0.0000 	0.0000 	-0.3740
""")

# mol = pynof.molecule("""
# 0 1
#   O  0.0000   0.000   0.116
#   H  0.0000   0.749  -0.453
#   H  0.0000  -0.749  -0.453
# """)

p = pynof.param(mol,"cc-pvdz")

p.ipnof = 5

p.RI = True

# E_jax = pynof.compute_energy_jax(mol,p)
# gamma,J_MO,K_MO,H_core,p
# E_t,C,n,fmiug0,grad= pynof.compute_energy(mol,p,gradients=True)

cj12,ck12,gamma,J_MO,K_MO,H_core,grad = pynof.compute_energy(mol,p,gradients=True)

calce_jax = lambda n, cj12, ck12, J_MO, K_MO, H_core, p: calce(n, cj12, ck12, J_MO, K_MO, H_core, p)


grad_calce_lambda = jax.grad(calce_jax,argnums=0)

def compute_grad(gamma, J_MO, K_MO, H_core, p):
    n, _ = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    cj12, ck12 = PNOFi_selector(n, p)
    return grad_calce_lambda(n, cj12, ck12, J_MO, K_MO, H_core, p)

grad_jax = compute_grad(gamma, J_MO, K_MO, H_core, p)
print(grad_jax)


hessian_calce_lambda = jax.hessian(calce_jax, argnums=0)


def compute_hessian(gamma, J_MO, K_MO, H_core, p):
    n, _ = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    cj12, ck12 = PNOFi_selector(n, p)
    return hessian_calce_lambda(n, cj12, ck12, J_MO, K_MO, H_core, p)

# Compute the Hessian
hessian_jax = compute_hessian(gamma, J_MO, K_MO, H_core, p)
print("Hessian:", hessian_jax)


def calce_wrapper(gamma, cj12, ck12, J_MO, K_MO, H_core, p):
    n, _ = jax_ocupacion_softmax(gamma, p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
    return calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

# print('gamma',gamma)
grad_calce_wrt_gamma = jax.grad(calce_wrapper, argnums=0)

# Compute the gradient
gamma_grad = grad_calce_wrt_gamma(gamma, cj12, ck12, J_MO, K_MO, H_core, p)
print("Gradient of calce wrt gamma:", gamma_grad)

print('gamma',gamma)
# eigenvalues = jnp.linalg.eigvals(gamma_grad)

# Print the eigenvalues
# print("Eigenvalues:", eigenvalues)

# Check if all eigenvalues are non-negative
# is_positive_semi_definite = jnp.all(eigenvalues >= 0)
# print("Is the matrix positive semi-definite?", is_positive_semi_definite)