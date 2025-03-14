import pynof
import jax
import jax.numpy as jnp 
import numpy as np
from jax import config
from pynof.alex_pynof import alex_CJCKD5,alex_calce,test,alex_calcoccg,alex_calcocce

config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)

mol = pynof.molecule("""
0 1
  H  0.0000 	0.0000 	0.0000
  H  0.0000 	0.0000 	0.7000
""")


p = pynof.param(mol,"6-31G")

p.ipnof = 5
# p.set_ncwo(2)

p.RI = True

cj12,ck12,gamma,J_MO,K_MO,H_core,n = pynof.compute_energy(mol,p,gradients=True)
print(n)
# n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, 'Softmax')
