import pynof
from pynof.pnof import calce,calcoccg

mol = pynof.molecule("""
0 1
  H  0.0000 	0.0000 	-0.9000
  H  0.0000 	0.0000 	0.9000
""")

p = pynof.param(mol,"cc-pvdz")

p.ipnof = 5
p.set_ncwo(2)

p.RI = True

# E,C,n,fmiug0 = pynof.compute_energy(mol,p)
cj12,ck12,gamma,J_MO,K_MO,H_core,grad = pynof.compute_energy(mol,p,gradients=True)
# print('Calce',calce(gamma,cj12, ck12,J_MO,K_MO,H_core,p))
print(calcoccg(gamma,J_MO,K_MO,H_core,p))