import pynof
import jax
import jax.numpy as jnp 
from jax import config

from jax import jacfwd, jacrev
config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)

from pynof.alex_pynof import alex_ocupacion_softmax,alex_der_CJCKD5_second_derivatives,alex_der_CJCKD5
from pynof.pnof import PNOFi_selector, ocupacion,der_PNOFi_selector

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
dn2_d2gamma_ = lambda gamma : alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
d2n_d2gamma = jax.hessian(dn2_d2gamma_,argnums=0)
d2n_d2gamma = d2n_d2gamma(gamma)[0]
dn_dgamma_ = lambda gamma : alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)

dn_dgamma = jax.jacobian(dn_dgamma_,argnums=0)
dn_dgamma = dn_dgamma(gamma)[0]
n, _ = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)


# print(der_CJCKD5_second_derivatives(n,dn_dgamma, d2n_d2gamma, p.no1, p.ndoc, p.nv, p.nbf5, p.ndns, p.ncwo))
# print('jax_dn_dgamma',dn_dgamma(gamma)[0])

def calcoccg_hessian(gamma, J_MO, K_MO, H_core, p):
    """
    Compute the Hessian of the energy with respect to gamma.
    """
    grad = jnp.zeros((p.nv))
    dn_dgamma = jax.jacobian(dn_dgamma_,argnums=0)
    dn_dgamma = dn_dgamma(gamma)[0]
    
    
    dn2_d2gamma_ = lambda gamma : alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
    d2n_d2gamma = jax.hessian(dn2_d2gamma_,argnums=0)
    d2n_d2gamma = d2n_d2gamma(gamma)[0]

    Dcj12r, Dck12r = alex_der_CJCKD5(n, p.ista, dn_dgamma, p.no1, p.ndoc, 
                                     p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo)
    Dcj12r, Dck12r = jnp.asarray(Dcj12r), jnp.asarray(Dck12r)
    
    # grad += jnp.einsum('ikl,i->kl',d2n_d2gamma,2*H_core+jnp.diagonal(J_MO),optimize=True)
    # print(grad)
    # assert 0 
    # 2 dCJ_dgamma J_MO
    # Dcj12r = Dcj12r.at[jnp.diag_indices(p.nbf5)].set(0)

    # grad += 2 * jnp.einsum('ijk,ji->k', Dcj12r, J_MO, optimize=True)
    
    # Dck12r = Dck12r.at[jnp.diag_indices(p.nbf5)].set(0)
    # grad -= 2 * jnp.einsum('ijk,ji->k', Dck12r, K_MO, optimize=True)
    
    # Compute second derivatives of Dcj12r and Dck12r with respect to gamma
    Dcj12r_second_deriv, Dck12r_second_deriv = alex_der_CJCKD5_second_derivatives(n,dn_dgamma, d2n_d2gamma, p.no1, p.ndoc, p.nv, p.nbf5, p.ndns, p.ncwo)
    # Dcj12r_second_deriv = Dcj12r_second_deriv.at[jnp.diag_indices(p.nbf5)].set(0)
    # Dck12r_second_deriv = Dck12r_second_deriv.at[jnp.diag_indices(p.nbf5)].set(0)
    # # Initialize Hessian
    hessian = jnp.zeros((p.nv, p.nv))
    
    # # Contribution from dn_dgamma term
    hessian += jnp.einsum('ikl,i->kl', d2n_d2gamma, 2 * H_core + jnp.diagonal(J_MO), optimize=True)
    
    # # Contribution from Dcj12r term
    hessian += 2 * jnp.einsum('ijkl,ji->kl', Dcj12r_second_deriv, J_MO, optimize=True)
    
    # # Contribution from Dck12r term
    hessian -= 2 * jnp.einsum('ijkl,ji->kl', Dck12r_second_deriv, K_MO, optimize=True)
    
    return hessian

hessian_wrt_gamma = calcoccg_hessian(gamma, J_MO, K_MO, H_core, p)
print(hessian_wrt_gamma)

eigenvalues = jnp.linalg.eigvals(hessian_wrt_gamma)

# Print the eigenvalues
print("Eigenvalues:", eigenvalues)

# Check if all eigenvalues are non-negative
is_positive_semi_definite = jnp.all(eigenvalues >= 0)
print("Is the matrix positive semi-definite?", is_positive_semi_definite)