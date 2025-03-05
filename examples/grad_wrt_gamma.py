import pynof
import jax
import jax.numpy as jnp 
from jax import config

from jax import jacfwd, jacrev
config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)

from pynof.alex_pynof import alex_calcocce,alex_calce,alex_calcoccg,alex_calcoccg_jax,alex_ocupacion_softmax
from pynof.pnof import PNOFi_selector, ocupacion,der_PNOFi_selector,calce

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

pynof_grad = pynof.calcoccg(gamma,J_MO,K_MO,H_core,p)
print('pynof_grad',pynof_grad)

def calce_wrapper(gamma, cj12, ck12, J_MO, K_MO, H_core, p):
    n, _ = alex_ocupacion_softmax(gamma, p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
    # jax.debug.print("Alex pynof n-> {x}", x=n)
    return alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

# print('gammas',gamma)
grad_calce_wrt_gamma = jax.grad(calce_wrapper, argnums=0)

# Compute the gradient
gamma_grad = grad_calce_wrt_gamma(gamma, cj12, ck12, J_MO, K_MO, H_core, p)
print("Gradient of calce wrt gamma:", gamma_grad)

analytical_grad = alex_calcoccg(gamma, J_MO, K_MO, H_core, p)
print("Analytical gradient:", analytical_grad)

# # Compute analytical gradient
# analytical_grad = alex_calcoccg(gamma, J_MO, K_MO, H_core, p)

# # Compute AD gradient
# def ad_wrapper(gamma):
#     return calce_wrapper(gamma, cj12, ck12, J_MO, K_MO, H_core, p)
# ad_grad = jax.grad(ad_wrapper)(gamma)

# # Compare results
# print("Analytical gradient:", analytical_grad)
# print("AD gradient:", ad_grad)
# print("Difference:", analytical_grad - ad_grad)

# # Check intermediate results
# n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
# Dcj12r, Dck12r = der_PNOFi_selector(n, dn_dgamma, p)

# print("n:", n)
# print("dn_dgamma:", dn_dgamma)
# print("Dcj12r:", Dcj12r)
# print("Dck12r:", Dck12r)

# # Compute ∂E/∂n manually
# dE_dn = 2 * H_core + jnp.diagonal(J_MO)
# print("∂E/∂n (manual):", dE_dn)

# # Compute ∂E/∂γ manually
# manual_grad = jnp.einsum('ik,i->k', dn_dgamma, dE_dn, optimize=True)
# print("∂E/∂γ (manual):", manual_grad)