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

# mol = pynof.molecule("""
# 0 1
#   H  0.0000 	0.0000 	0.0000
#   H  0.0000 	0.0000 	0.7000
# """)


mol = pynof.molecule("""
0 1
  O  0.0000   0.000   0.116
  H  0.0000   0.749  -0.453
  H  0.0000  -0.749  -0.453
""")
p = pynof.param(mol,"cc-pvdz")

p.ipnof = 5
# p.set_ncwo(10)

# p.RI = True

#Using pynof to print energy 
E = lambda n, cj12, ck12, J_MO, K_MO, H_core, p: calce(n, cj12, ck12, J_MO, K_MO, H_core, p)


cj12,ck12,gamma,J_MO,K_MO,H_core,grad = pynof.compute_energy(mol,p,gradients=True)


assert 0 

#Using alex pynof to calculate the jax.grad of the Energy wrt n 
calce_jax = lambda n, cj12, ck12, J_MO, K_MO, H_core, p: alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

grad_calce_lambda = jax.grad(calce_jax,argnums=0)

def compute_grad(gamma, J_MO, K_MO, H_core, p):
    # n, _ = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    n, _ = alex_ocupacion_softmax(gamma, p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
    cj12, ck12 = PNOFi_selector(n, p)
    return grad_calce_lambda(n, cj12, ck12, J_MO, K_MO, H_core, p)

grad_jax = compute_grad(gamma, J_MO, K_MO, H_core, p)
print('grad_jax',grad_jax)


assert 0 

dn_dgamma_ = lambda gamma : alex_ocupacion_softmax(gamma,p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)

dn_dgamma = jax.jacobian(dn_dgamma_,argnums=0)

# print('jax_dn_dgamma',dn_dgamma(gamma)[0])
# assert 0

hessian_calce_lambda = jax.hessian(calce_jax, argnums=0)

# assert 0

def compute_hessian(gamma, J_MO, K_MO, H_core, p):
    n, _ = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    cj12, ck12 = PNOFi_selector(n, p)
    return hessian_calce_lambda(n, cj12, ck12, J_MO, K_MO, H_core, p)

# Compute the Hessian
hessian_jax = compute_hessian(gamma, J_MO, K_MO, H_core, p)
# print("Hessian:", hessian_jax)


def calce_wrapper(gamma, cj12, ck12, J_MO, K_MO, H_core, p):
    n, _ = alex_ocupacion_softmax(gamma, p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
    # print('n_jax',n)
    return alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

# print('gammas',gamma)
grad_calce_wrt_gamma = jax.grad(calce_wrapper, argnums=0)

# Compute the gradient
gamma_grad = grad_calce_wrt_gamma(gamma, cj12, ck12, J_MO, K_MO, H_core, p)
print("Gradient of calce wrt gamma:", gamma_grad)

pynof_grad = pynof.calcoccg(gamma,J_MO,K_MO,H_core,p)
print('pynof_grad',pynof_grad)

# alex_jax_grad = calcoccg(gamma,J_MO,K_MO,H_core,p)
# print('alex_grad',alex_jax_grad)

assert 0 
# print('gamma',gamma)
# gamma [ -0.96457745  -5.59726465  -6.74907508  -7.37209657  -7.37209693
#  -10.17475865 -10.39184356 -10.39184833 -10.43887492 -12.49338305]
# assert 0 
calcocce_jax = lambda gamma: alex_calcocce(gamma,cj12, ck12,J_MO,K_MO,H_core,p)
grad_calcocce_jax = jax.grad(calcocce_jax,argnums=0)
print(grad_calcocce_jax(gamma))


def finite_difference_gradient(f, x, epsilon=1e-10):
    """
    Compute the gradient of a function f using finite differences.

    Args:
        f: A function that takes a single argument (x) and returns a scalar.
        x: The input at which to compute the gradient.
        epsilon: The perturbation size for finite differences.

    Returns:
        grad_fd: The finite difference approximation of the gradient.
    """
    grad_fd = jnp.zeros_like(x)
    for i in range(x.size):
        x_plus = x.at[i].add(epsilon)  # Perturb x[i] by +epsilon
        x_minus = x.at[i].add(-epsilon)  # Perturb x[i] by -epsilon
        f_plus = f(x_plus)  # Evaluate f at x_plus
        f_minus = f(x_minus)  # Evaluate f at x_minus
        grad_fd = grad_fd.at[i].set((f_plus - f_minus) / (2 * epsilon))  # Central difference
    return grad_fd


def energy(gamma, J_MO, K_MO, H_core, p):
    n, _ = alex_ocupacion_softmax(gamma, p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
    # print('n_jax',n)
    return alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

# Wrap the energy function to depend only on gamma
def energy_wrapper(gamma):
    return energy(gamma, J_MO, K_MO, H_core, p)

n, _ = alex_ocupacion_softmax(gamma, p.no1, p.ndoc, p.nalpha, p.nv, 
                                 p.nbf5, p.ndns, p.ncwo, p.HighSpin)
# Compute the gradient using finite differences
gamma = jnp.array(gamma)  # Convert gamma to a JAX array
grad_fd = finite_difference_gradient(energy_wrapper, gamma)

# Compute the gradient using JAX's automatic differentiation
grad_ad = jax.grad(energy_wrapper)(gamma)

# Compare the results
print('Finite difference gradient:', grad_fd)
print('Automatic differentiation gradient:', grad_ad)
print('Difference:', jnp.linalg.norm(grad_fd - grad_ad))
# print('gamma',gamma)
# eigenvalues = jnp.linalg.eigvals(gamma_grad)

# Print the eigenvalues
# print("Eigenvalues:", eigenvalues)

# Check if all eigenvalues are non-negative
# is_positive_semi_definite = jnp.all(eigenvalues >= 0)
# print("Is the matrix positive semi-definite?", is_positive_semi_definite)