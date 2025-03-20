import numpy as np
from scipy.linalg import eigh
from time import time
import pynof
import psi4
import jax 
import jax.numpy as jnp
from pynof.alex_pynof import alex_calce
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def occoptr_conv(gamma,C,H,I,b_mnl,p):
    J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)
    E = 0
    nit = 0
    success = True
    xs = []
    def callback(xk):
        xs.append(np.copy(xk))
    if (p.ndoc>0):
        res = minimize(pynof.calcocce, gamma, args=(J_MO,K_MO,H_core,p), jac=pynof.calcoccg, method=p.occupation_optimizer, callback=callback)
        gamma = res.x
        E = res.fun
        nit = res.nit
        success = res.success
    n,dR = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
    cj12,ck12 = pynof.PNOFi_selector(n,p)
    return E,nit,success,gamma,n,cj12,ck12,xs

def compute_energy_conv(mol,p=None,C=None,n=None,fmiug0=None,guess="HF",nofmp2=False,mbpt=False, gradients=False,printmode=True,ekt=False,mulliken_pop=False,lowdin_pop=False,m_diagnostic=False,perturb=False,erpa=False,iter_erpa=0):
    """Compute Natural Orbital Functional single point energy"""
    t1 = time()    
    wfn = p.wfn
    S,T,V,H,I,b_mnl,Dipole = pynof.compute_integrals(wfn,mol,p)

    # Temporary lists
    Cs = []
    xss = []
    Es_t = []

    if(printmode):
        print("Number of basis functions                   (NBF)    =",p.nbf)
        if(p.RI):
            print("Number of auxiliary basis functions         (NBFAUX) =",p.nbfaux)
        print("Inactive Doubly occupied orbitals up to     (NO1)    =",p.no1)
        print("No. considered Strongly Doubly occupied MOs (NDOC)   =",p.ndoc)
        print("No. considered Strongly Singly occupied MOs (NSOC)   =",p.nsoc)
        print("No. of Weakly occ. per St. Doubly occ.  MOs (NCWO)   =",p.ncwo)
        print("Dimension of the Nat. Orb. subspace         (NBF5)   =",p.nbf5)
        print("No. of electrons                                     =",p.ne)
        print("Multiplicity                                         =",p.mul)
        print("")

    # Nuclear Energy
    E_nuc = mol.nuclear_repulsion_energy()

    # Guess de MO (C)
    if(C is None):
        if guess=="Core" or guess==None:
            Eguess,C = eigh(H, S)  # (HC = SCe)
        elif guess=="HFIDr":
            Eguess,Cguess = eigh(H, S)
            EHF,C,fmiug0guess = pynof.hfidr(Cguess,H,I,b_mnl,E_nuc,p,printmode)
        else:
            EHF, wfn_HF = psi4.energy(guess, return_wfn=True)
            EHF = EHF - E_nuc
            C = wfn_HF.Ca().np
    else:
        guess = None
        C_old = np.copy(C)
        for i in range(p.ndoc):
            for j in range(p.ncwo):
                k = p.no1 + p.ndns + (p.ndoc - i - 1) * p.ncwo + j
                l = p.no1 + p.ndns + (p.ndoc - i - 1) + j*p.ndoc
                C[:,k] = C_old[:,l]
    
    C = pynof.check_ortho(C,S,p)
    Cs.append(C)
    # Guess Occupation Numbers (n)
    if(n is None):
        if p.occ_method == "Trigonometric":
            gamma = pynof.compute_gammas_trigonometric(p.ndoc,p.ncwo)
        if p.occ_method == "Softmax":
            p.nv = p.nbf5 - p.no1 - p.nsoc 
            gamma = pynof.compute_gammas_softmax(p.ndoc,p.ncwo)
        if p.occ_method == "EBI":
            p.nbf5 = p.nbf
            p.nvar = int(p.nbf*(p.nbf-1)/2)
            p.nv = p.nbf
            gamma = pynof.compute_gammas_ebi(p.ndoc,p.nbf)
    else:
        n_old = np.copy(n)
        for i in range(p.ndoc):
            for j in range(p.ncwo):
                k = p.no1 + p.ndns + (p.ndoc - i - 1) * p.ncwo + j
                l = p.no1 + p.ndns + (p.ndoc - i - 1) + j*p.ndoc
                n[k] = n_old[l]
        if p.occ_method == "Trigonometric":
            gamma = pynof.n_to_gammas_trigonometric(n,p.no1,p.ndoc,p.ndns,p.ncwo)
        if p.occ_method == "Softmax":
            p.nv = p.nbf5 - p.no1 - p.nsoc 
            gamma = pynof.n_to_gammas_softmax(n,p.no1,p.ndoc,p.ndns,p.ncwo)
        if p.occ_method == "EBI":
            p.nbf5 = p.nbf
            p.nvar = int(p.nbf*(p.nbf-1)/2)
            p.nv = p.nbf
            gamma = pynof.n_to_gammas_ebi(n)
    
    
    elag = np.zeros((p.nbf,p.nbf))
    
    E_occ,nit_occ,success_occ,gamma,n,cj12,ck12,xs = occoptr_conv(gamma,C,H,I,b_mnl,p)
    xss.append(xs)

    # COMPUTE THE ENERGY - THIS IS NOT IN THE PYNOF CODE 
   
    #if(p.orb_method=="ID"):
    #    E_orb,C,nit_orb,success_orb,itlim,fmiug0 = pynof.orboptr(C,n,H,I,b_mnl,cj12,ck12,i_ext,itlim,fmiug0,p,printmode)
    if(p.orb_method=="Rotations"):
        E_orb,C,nit_orb,success_orb = pynof.orbopt_rotations(gamma,C,H,I,b_mnl,p)
    if(p.orb_method=="ADAM"):
        E_orb,C,nit_orb,success_orb = pynof.orbopt_adam(gamma,C,H,I,b_mnl,p)
    
    Es_t.append(E_orb)

    iloop = 0
    itlim = 0
    E,E_old,E_diff = 9999,9999,9999
    Estored,Cstored,gammastored = 0,0,0
    last_iter = 0

    if(printmode):
        print("")
        print("PNOF{} Calculation ({}/{} Optimization)".format(p.ipnof,p.orb_method,p.occ_method))
        print("==================")
        print("")
        print('{:^7} {:^7}  {:^7}  {:^14} {:^14} {:^14}   {:^6}   {:^6} {:^6} {:^6}'.format("Nitext","Nit_orb","Nit_occ","Eelec","Etot","Ediff","Grad_orb","Grad_occ","Conv Orb","Conv Occ"))
    for i_ext in range(p.maxit):
        #orboptr
        #t1 = time()
        if(p.orb_method=="ID"):
            E_orb,C,nit_orb,success_orb,itlim,fmiug0 = pynof.orboptr(C,n,H,I,b_mnl,cj12,ck12,i_ext,itlim,fmiug0,p,printmode)
        if(p.orb_method=="Rotations"):
            E_orb,C,nit_orb,success_orb = pynof.orbopt_rotations(gamma,C,H,I,b_mnl,p)
        if(p.orb_method=="ADAM"):
            E_orb,C,nit_orb,success_orb = pynof.orbopt_adam(gamma,C,H,I,b_mnl,p)
        #t2 = time()
        
        #occopt
        E_occ,nit_occ,success_occ,gamma,n,cj12,ck12,xs = occoptr_conv(gamma,C,H,I,b_mnl,p)

        if(p.occ_method=="Softmax"):
            C,gamma = pynof.order_occupations_softmax(C,gamma,H,I,b_mnl,p)

        E = E_orb
        E_diff = E-E_old
        E_old = E

        y = np.zeros((p.nvar))
        grad_orb = pynof.calcorbg(y,n,cj12,ck12,C,H,I,b_mnl,p)
        J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)
        
        
        grad_occ = pynof.calcoccg(gamma,J_MO,K_MO,H_core,p)

        print("{:6d} {:6d} {:6d}   {:14.8f} {:14.8f} {:15.8f}      {:3.1e}    {:3.1e}   {}   {}".format(i_ext,nit_orb,nit_occ,E,E+E_nuc,E_diff,np.linalg.norm(grad_orb),np.linalg.norm(grad_occ),success_orb,success_occ))
    
        if(success_orb or (np.linalg.norm(grad_orb) < 1e-3 and np.linalg.norm(grad_occ)< 1e-3)):

            if perturb and E - Estored < -1e-4:
                y = np.zeros((p.nvar))
                grad_orb = pynof.calcorbg(y,n,cj12,ck12,C,H,I,b_mnl,p)
                J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)
                grad_occ = pynof.calcoccg(gamma,J_MO,K_MO,H_core,p)
                print("Increasing Gradient")
                last_iter = i_ext
                Estored,Cstored,gammastored = E,C.copy(),gamma.copy()
                C,gamma = pynof.perturb_solution(C,gamma,grad_orb,grad_occ,p)
            else:
                print("Solution does not improve anymore")
                if(Estored<E):
                    E,C,gamma = Estored,Cstored,gammastored
                break
    
        n,dR = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
        cj12,ck12 = pynof.PNOFi_selector(n,p)
        E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)
        print("\nLagrage sumdiff {:3.1e} maxfdiff {:3.1e}".format(sumdiff,maxdiff))

        if(p.ipnof>4):
            C,n,elag = pynof.order_subspaces(C,n,elag,H,I,b_mnl,p)

        C_old = np.copy(C)
        n_old = np.copy(n)
        for i in range(p.ndoc):
            for j in range(p.ncwo):
                k = p.no1 + p.ndns + (p.ndoc - i - 1) * p.ncwo + j
                l = p.no1 + p.ndns + (p.ndoc - i - 1) + j*p.ndoc
                C[:,l] = C_old[:,k]
                n[l] = n_old[k]

        E_t = E_nuc + E

        xss.append(xs)
        Cs.append(C)
        Es_t.append(E_t)

    return J_MO, K_MO, H_core, xss, Cs, Es_t

def spd_f(H):

    eig_values_H = jnp.linalg.eigvals(H)
    return jnp.all(eig_values_H.real >= -1e-10)

def E_step(J_MO,K_MO,H_core,xs,p):

    Es_n = []
    results = []

    for i,x in enumerate(xs):

        n, _ = pynof.ocupacion(x, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, 'Softmax')

        cj12, ck12 = pynof.PNOFi_selector(n,p)

        E_n = lambda n: alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

        grad_E_n = jax.jacfwd(E_n,argnums=0)
        hessian = jax.jacrev(grad_E_n,argnums=0)
        result = spd_f(hessian(n))

        Es_n.append(E_n(n))
        results.append(result)

    return np.array(Es_n), np.array(results)

# def plots(J_MO,K_MO,H_core,xss,Cs,Es_t,p):
# #This one plots all in the same one, looks bad
#     for i,C in enumerate(Cs):
#         Es_n, results = E_step(J_MO,K_MO,H_core,xss[i],p)
#         colors = np.where(results, 'green', 'red')
    
#         plt.plot(range(len(Es_n)), Es_n, color='black', linestyle='-', linewidth=2)
#         plt.scatter(range(len(Es_n)), Es_n, c=colors, edgecolor='black',s=200)
        
#         plt.xlabel("Iteration",size=25)
#         plt.ylabel("Energy (Ha)",size=25)
#         plt.xticks(fontsize=16)
#         plt.yticks(fontsize=16)
#         plt.plot(Es_n)
#         filename = f"graph_{i}.png"
#         plt.savefig(filename, format='png', dpi=600, bbox_inches='tight')

def plots_joint(J_MO, K_MO, H_core, xss, Cs, Es_t, p):
    plt.figure(figsize=(12, 8))  # Create a single figure

    total_iterations = 0  # Track x-axis shifts for each graph

    for i, C in enumerate(Cs):
        Es_n, results = E_step(J_MO, K_MO, H_core, xss[i], p)
        colors = np.where(results, 'green', 'red')
        
        x_vals = range(total_iterations, total_iterations + len(Es_n))  # Adjust x-axis to continue from last
        plt.plot(x_vals, Es_n, color='black', linestyle='-', linewidth=2)
        plt.scatter(x_vals, Es_n, c=colors, edgecolor='black', s=100)
        
        total_iterations += len(Es_n)  # Update x-axis shift for next plot
    
    # Formatting
    plt.xlabel("Iteration", size=25, labelpad=10)
    plt.ylabel("Energy (Ha)", size=25, labelpad=10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    filename = "concatenated_graph.png"
    plt.savefig(filename, format='png', dpi=600, bbox_inches='tight')
    

def plots(J_MO, K_MO, H_core, xss, Cs, Es_t, p):
#This one saves each iteration on a new file
    for i, C in enumerate(Cs):
        Es_n, results = E_step(J_MO, K_MO, H_core, xss[i], p)
        colors = np.where(results, 'green', 'red')
        
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(Es_n)), Es_n, color='black', linestyle='-', linewidth=2)
        plt.scatter(range(len(Es_n)), Es_n, c=colors, edgecolor='black', s=200)
        
        plt.xlabel("Iteration", size=25)
        plt.ylabel("Energy (Ha)", size=25)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        filename = f"graph_{i}.png"
        plt.savefig(filename, format='png', dpi=600,bbox_inches='tight')
        plt.close()  # Close the figure to free memory

