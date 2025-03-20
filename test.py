import matplotlib.pyplot as plt
import pynof
import jax
import jax.numpy as jnp
from jax import config
from jax import jacfwd, jacrev
config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)
import numpy as np
from pynof.alex_pynof import alex_calce #alex_CJCKD5,alex_calce,test,alex_calcoccg,alex_calcocce
from pynof.pnof import PNOFi_selector,ocupacion,der_PNOFi_selector,calce,calcoccg,calcocce
from energy import *

mol = pynof.molecule("""
0 1
  H  0.0000     0.0000  0.0000
  H  0.0000     0.0000  1.1000
""")
p = pynof.param(mol,'aug-cc-pvdz')#"6-31g" 'aug-cc-pvdz')
p.ipnof = 5
#p.set_ncwo(1)
p.RI = True
p.occ_method = "Trigonometric"
p.orb_method=="ADAM"
J_MO, K_MO, H_core, xss, Cs, Es_t = compute_energy_conv(mol,p,C=None,guess=None)
plots(J_MO,K_MO,H_core,xss,Cs,Es_t,p)
plots_joint(J_MO,K_MO,H_core,xss,Cs,Es_t,p)
