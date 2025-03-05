import jax 
import jax.numpy as jnp 
from pynof.pnof import PNOFi_selector, ocupacion,der_PNOFi_selector,calce

from scipy.differentiate import hessian

from jax import lax
import numpy as np

jax.config.update("jax_enable_x64", True)

def alex_JKH_MO_RI(C,H,b_mnl,p):

    #denmatj
    D = jnp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    #b transform
    b_pnl = jnp.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_pql = jnp.einsum('nq,pnl->pql',C[:,0:p.nbf5],b_pnl, optimize=True)
    #QJMATm
    J_MO = jnp.einsum('ppl,qql->pq', b_pql, b_pql, optimize=True)
    #QKMATm
    K_MO = jnp.einsum('pql,pql->pq', b_pql, b_pql, optimize=True)
    #QHMATm
    H_core = jnp.tensordot(D,H, axes=([1,2],[0,1]))

    return H_core

def alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p):
    E = 0
    n = jnp.asarray(n)
    # print(n)
    # cj12 = jnp.asarray(cj12)
    # ck12 = jnp.asarray(ck12)
    cj12, ck12 = alex_CJCKD5(n, p.no1, p.ndoc, p.nsoc, 
                p.nbeta, p.nalpha, p.ndns, p.ncwo,p.MSpin)
    
    # jax.debug.print('cj12_j{x}',x=cj12)
    # jax.debug.print('ck12_j{y}',y=ck12)
    if p.MSpin == 0:
        E += 2 * jnp.einsum('i,i', n, H_core, optimize=True)
       
        E += jnp.einsum('i,i', n[:p.nbeta], jnp.diagonal(J_MO)[:p.nbeta], optimize=True)
       
        E += jnp.einsum('i,i', n[p.nalpha:p.nbf5], jnp.diagonal(J_MO)[p.nalpha:p.nbf5], optimize=True)
        
        cj12 = cj12.at[jnp.diag_indices_from(cj12)].set(0)
        E += jnp.einsum('ij,ji->', cj12, J_MO, optimize=True)
        # jax.debug.print("cj12 term: {E_cj12}", E_cj12=cj12)
        
        ck12 = ck12.at[jnp.diag_indices_from(ck12)].set(0)
        E -= jnp.einsum('ij,ji->', ck12, K_MO, optimize=True)
        # jax.debug.print("Alex pynof E-> {x}", x=E)
        # jax.debug.print("ck12 term: {E_ck12}", E_ck12=ck12)
        # ck12 = ck12.at[jnp.diag_indices_from(ck12)].set(0)
        # E_ck12 = jnp.einsum('ij,ji->', ck12, K_MO, optimize=True)
        # grad_ck12 = jax.grad(lambda n: jnp.einsum('ij,ji->', ck12, K_MO, optimize=True))(n)
        # jax.debug.print("Gradient of ck12 term: {grad_ck12}", grad_ck12=grad_ck12)
        # E -= E_ck12
    else:
        E += jnp.einsum('i,i', n[:p.nbeta], 2 * H_core[:p.nbeta] + jnp.diagonal(J_MO)[:p.nbeta], optimize=True)
        E += jnp.einsum('i,i', n[p.nbeta:p.nalpha], 2 * H_core[p.nbeta:p.nalpha], optimize=True)
        E += jnp.einsum('i,i', n[p.nalpha:p.nbf5], 2 * H_core[p.nalpha:p.nbf5] + jnp.diagonal(J_MO)[p.nalpha:p.nbf5], optimize=True)

        cj12 = cj12.at[jnp.diag_indices_from(cj12)].set(0)
        E += jnp.einsum('ij,ji->', cj12[:p.nbeta, :p.nbeta], J_MO[:p.nbeta, :p.nbeta], optimize=True)
        E += jnp.einsum('ij,ji->', cj12[:p.nbeta, p.nalpha:p.nbf5], J_MO[p.nalpha:p.nbf5, :p.nbeta], optimize=True)
        E += jnp.einsum('ij,ji->', cj12[p.nalpha:p.nbf5, :p.nbeta], J_MO[:p.nbeta, p.nalpha:p.nbf5], optimize=True)
        E += jnp.einsum('ij,ji->', cj12[p.nalpha:p.nbf5, p.nalpha:p.nbf5], J_MO[p.nalpha:p.nbf5, p.nalpha:p.nbf5], optimize=True)

        ck12 = ck12.at[jnp.diag_indices_from(ck12)].set(0)
        E -= jnp.einsum('ij,ji->', ck12[:p.nbeta, :p.nbeta], K_MO[:p.nbeta, :p.nbeta], optimize=True)
        E -= jnp.einsum('ij,ji->', ck12[:p.nbeta, p.nalpha:p.nbf5], K_MO[p.nalpha:p.nbf5, :p.nbeta], optimize=True)
        E -= jnp.einsum('ij,ji->', ck12[p.nalpha:p.nbf5, :p.nbeta], K_MO[:p.nbeta, p.nalpha:p.nbf5], optimize=True)
        E -= jnp.einsum('ij,ji->', ck12[p.nalpha:p.nbf5, p.nalpha:p.nbf5], K_MO[p.nalpha:p.nbf5, p.nalpha:p.nbf5], optimize=True)

        E += 2 * jnp.einsum('i,ji->', n[:p.nbeta], J_MO[p.nbeta:p.nalpha, :p.nbeta], optimize=True)
        E += 2 * jnp.einsum('i,ji->', n[p.nalpha:p.nbf5], J_MO[p.nbeta:p.nalpha, p.nalpha:p.nbf5], optimize=True)
        E += 0.5 * (jnp.einsum('i,ji->', n[p.nbeta:p.nalpha], J_MO[p.nbeta:p.nalpha, p.nbeta:p.nalpha], optimize=True) -
                    jnp.einsum('i,ii->', n[p.nbeta:p.nalpha], J_MO[p.nbeta:p.nalpha, p.nbeta:p.nalpha], optimize=True))

        E -= jnp.einsum('i,ji->', n[:p.nbeta], K_MO[p.nbeta:p.nalpha, :p.nbeta], optimize=True)
        E -= jnp.einsum('i,ji->', n[p.nalpha:p.nbf5], K_MO[p.nbeta:p.nalpha, p.nalpha:p.nbf5], optimize=True)
        E -= 0.5 * (jnp.einsum('i,ji->', n[p.nbeta:p.nalpha], K_MO[p.nbeta:p.nalpha, p.nbeta:p.nalpha], optimize=True) +
                    jnp.einsum('i,ii->', n[p.nbeta:p.nalpha], K_MO[p.nbeta:p.nalpha, p.nbeta:p.nalpha], optimize=True))
    # print('jax',E)
    # jax.debug.print("Alex pynof E-> {x}", x=E)
    return E

def test(n, cj12, ck12, J_MO, K_MO, H_core, p):
    E = 0.0
    n = jnp.asarray(n)
    
    # Compute cj12 and ck12
    cj12, ck12 = alex_CJCKD5(n, p.no1, p.ndoc, p.nsoc, 
                              p.nbeta, p.nalpha, p.ndns, p.ncwo, p.MSpin)
    
    # Linear terms (H_core and diagonal J_MO terms)
    E_linear = (
        2 * jnp.einsum('i,i', n, H_core, optimize=True) +
        jnp.einsum('i,i', n[:p.nbeta], jnp.diagonal(J_MO)[:p.nbeta], optimize=True) +
        jnp.einsum('i,i', n[p.nalpha:p.nbf5], jnp.diagonal(J_MO)[p.nalpha:p.nbf5], optimize=True)
    )
    
    # Non-linear terms (cj12 and ck12 contributions)
    E_cj12 = jnp.einsum('ij,ji->', cj12, J_MO, optimize=True)
    E_ck12 = jnp.einsum('ij,ji->', ck12, K_MO, optimize=True)
    
    # Total energy
    E = E_linear + E_cj12 - E_ck12
    
    return E, E_cj12, E_cj12


def alex_calcocce(gamma,cj12, ck12,J_MO,K_MO,H_core,p):

    # n,dn_dgamma = ocupation_trigonometric(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    # cj12,ck12 = PNOFi_selector(n,p)
    #cj12,ck12 = CJCKD5(n, p.no1, p.ndoc, p.nsoc, p.nbeta, p.nalpha, p.ndns, p.ncwo, p.MSpin)
    # cj12,ck12 = PNOFi_selector(n,p)

    # n = jnp.asarray(n)
    # cj12 = jnp.asarray(cj12)
    # ck12 = jnp.asarray(ck12)
    n,_ = alex_ocupacion_softmax(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    # print('jax_occ numbers',n)
    E = alex_calce(n,cj12,ck12,J_MO,K_MO,H_core,p)
    
    
    return E

def alex_ocupation_trigonometric(gamma, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin):
    """Transform gammas to n according to the trigonometric 
    parameterization of the occupation numbers using JAX."""

    n = jnp.zeros(nbf5)
    dn_dgamma = jnp.zeros((nbf5, nv))
    dni_dgammai = jnp.zeros(nbf5)

    # [1, no1]
    n = n.at[:no1].set(1.0)

    # (no1, no1+ndoc]
    n = n.at[no1:no1+ndoc].set(0.5 * (1 + jnp.cos(gamma[:ndoc])**2))
    dni_dgammai = dni_dgammai.at[no1:no1+ndoc].set(-0.5 * jnp.sin(2 * gamma[:ndoc]))

    for i in range(ndoc):
        # dn_g/dgamma_g
        dn_dgamma = dn_dgamma.at[no1+i, i].set(dni_dgammai[no1+i])

    if not HighSpin:
        n = n.at[no1+ndoc:no1+ndns].set(0.5)  # (no1+ndoc, no1+ndns]
    else:
        n = n.at[no1+ndoc:no1+ndns].set(1.0)  # (no1+ndoc, no1+ndns]

    h = 1 - n
    for i in range(ndoc):
        ll_n = no1 + ndns + (ndoc - i - 1) * ncwo
        ul_n = ll_n + ncwo
        n_pi = n[ll_n:ul_n]
        ll_gamma = ndoc + (ndoc - i - 1) * (ncwo - 1)
        ul_gamma = ll_gamma + (ncwo - 1)
        gamma_pi = gamma[ll_gamma:ul_gamma]

        # n_pi
        n_pi = n_pi.at[:].set(h[no1 + i])
        for kw in range(ncwo - 1):
            n_pi = n_pi.at[kw].set(n_pi[kw] * jnp.sin(gamma_pi[kw])**2)
            n_pi = n_pi.at[kw+1:].set(n_pi[kw+1:] * jnp.cos(gamma_pi[kw])**2)

        # dn_pi/dgamma_g
        dn_pi_dgamma_g = dn_dgamma[ll_n:ul_n, i]
        dn_pi_dgamma_g = dn_pi_dgamma_g.at[:].set(-dni_dgammai[no1 + i])
        for kw in range(ncwo - 1):
            dn_pi_dgamma_g = dn_pi_dgamma_g.at[kw].set(dn_pi_dgamma_g[kw] * jnp.sin(gamma_pi[kw])**2)
            dn_pi_dgamma_g = dn_pi_dgamma_g.at[kw+1:].set(dn_pi_dgamma_g[kw+1:] * jnp.cos(gamma_pi[kw])**2)

        # dn_pi/dgamma_pj (j < i)
        dn_pi_dgamma_pj = dn_dgamma[ll_n:ul_n, ll_gamma:ul_gamma]
        for jw in range(ncwo - 1):
            dn_pi_dgamma_pj = dn_pi_dgamma_pj.at[jw+1:, jw].set(n[no1 + i] - 1)
            for kw in range(jw):
                dn_pi_dgamma_pj = dn_pi_dgamma_pj.at[jw+1:, jw].set(dn_pi_dgamma_pj[jw+1:, jw] * jnp.cos(gamma_pi[kw])**2)
            dn_pi_dgamma_pj = dn_pi_dgamma_pj.at[jw+1:, jw].set(dn_pi_dgamma_pj[jw+1:, jw] * jnp.sin(2 * gamma_pi[jw]))
            for kw in range(jw + 1, ncwo - 1):
                dn_pi_dgamma_pj = dn_pi_dgamma_pj.at[kw, jw].set(dn_pi_dgamma_pj[kw, jw] * jnp.sin(gamma_pi[kw])**2)
                dn_pi_dgamma_pj = dn_pi_dgamma_pj.at[kw+1:, jw].set(dn_pi_dgamma_pj[kw+1:, jw] * jnp.cos(gamma_pi[kw])**2)

        # dn_pi/dgamma_i
        for jw in range(ncwo - 1):
            dn_pi_dgamma_pj = dn_pi_dgamma_pj.at[jw, jw].set(1 - n[no1 + i])
        for kw in range(ncwo - 1):
            dn_pi_dgamma_pj = dn_pi_dgamma_pj.at[kw, kw].set(dn_pi_dgamma_pj[kw, kw] * jnp.sin(2 * gamma_pi[kw]))
            for lw in range(kw + 1, ncwo - 1):
                dn_pi_dgamma_pj = dn_pi_dgamma_pj.at[lw, lw].set(dn_pi_dgamma_pj[lw, lw] * jnp.cos(gamma_pi[kw])**2)

    
    return n, dn_dgamma



def alex_calcoccg_jax(gamma, J_MO, K_MO, H_core, p):
    n,dn_dgamma = alex_ocupation_trigonometric(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin)
   
    # cj12, ck12 = der_PNOFi_selector(n, dn_dgamma, p)
    cj12,ck12 = PNOFi_selector(n,p)

    n = jnp.asarray(n)
    cj12 = jnp.asarray(cj12)
    ck12 = jnp.asarray(ck12)

    # Compute energy
    E = alex_calce(n, cj12, ck12, J_MO, K_MO, H_core, p)
    
    return E  # Return energy so JAX can differentiate it


def alex_calcoccg(gamma, J_MO, K_MO, H_core, p):
    grad = jnp.zeros((p.nv))
    # n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    n,dn_dgamma = alex_ocupacion_softmax(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    
    # Dcj12r, Dck12r = der_PNOFi_selector(n, dn_dgamma, p)
    Dcj12r, Dck12r = alex_der_CJCKD5(n, p.ista, dn_dgamma, p.no1, p.ndoc, 
                                     p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo,p)

    # Dcj12r, Dck12r = jnp.asarray(Dcj12r), jnp.asarray(Dck12r)
    # Compute ∂E/∂n
    dE_dn = 2 * H_core + jnp.diagonal(J_MO)
    
    # Compute ∂E/∂γ = ∂E/∂n · ∂n/∂γ
    grad += jnp.einsum('ik,i->k', dn_dgamma, dE_dn, optimize=True)
    
    # Add contributions from Dcj12r and Dck12r
    Dcj12r = Dcj12r.at[jnp.diag_indices(p.nbf5)].set(0)
    grad += 2 * jnp.einsum('ijk,ji->k', Dcj12r, J_MO, optimize=True)
    
    Dck12r = Dck12r.at[jnp.diag_indices(p.nbf5)].set(0)
    grad -= 2 * jnp.einsum('ijk,ji->k', Dck12r, K_MO, optimize=True)
    
    # print('alex_Dck',Dck12r)
    # print('alex_grad',grad)
    return grad

def alex_ocupacion_softmax(x, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin):
    """Transform gammas to n according to the softmax 
    parameterization of the occupation numbers"""

    n = jnp.zeros((nbf5))
    dn_dx = jnp.zeros((nbf5, nv))

    # Set initial values for n based on HighSpin
    if not HighSpin:
        n = n.at[no1+ndoc:no1+ndns].set(0.5)   # (no1+ndoc, no1+ndns]
    else:
        n = n.at[no1+ndoc:no1+ndns].set(1.0)   # (no1+ndoc, no1+ndns]

    exp_x = jnp.exp(x)

    def body_fun(i, carry):
        n, dn_dx = carry

        # Compute dynamic slice indices
        ll = no1 + ndns + (ndoc - i - 1) * ncwo
        ul = ll + ncwo

        # Use dynamic_slice to extract n_pi
        n_pi = lax.dynamic_slice(n, (ll,), (ncwo,))

        # Compute dynamic slice indices for x
        ll_x = ll - ndns + ndoc - no1
        ul_x = ll_x + ncwo

        # Extract relevant parts of dn_dx
        dn_pi_dx_pi = lax.dynamic_slice(dn_dx, (ll, ll_x), (ncwo, ncwo))
        dn_g_dx_pi = lax.dynamic_slice(dn_dx, (i, ll_x), (1, ncwo)).squeeze(0)
        dn_pi_dx_g = lax.dynamic_slice(dn_dx, (ll, i), (ncwo, 1)).squeeze(1)

        # Extract exp_x_pi
        exp_x_pi = lax.dynamic_slice(exp_x, (ll_x,), (ncwo,))

        # Compute sum_exp
        sum_exp = exp_x[i] + jnp.sum(exp_x_pi)

        # Update n[i] and n_pi
        n = n.at[i].set(exp_x[i] / sum_exp)
        n = lax.dynamic_update_slice(n, exp_x_pi / sum_exp, (ll,))

        # Update dn_pi_dx_pi
        outer_prod = -jnp.outer(exp_x_pi, exp_x_pi) / sum_exp**2
        dn_pi_dx_pi = outer_prod
        dn_dx = lax.dynamic_update_slice(dn_dx, dn_pi_dx_pi, (ll, ll_x))

        # Update dn_g_dx_pi and dn_pi_dx_g
        dn_g_dx_pi = -exp_x_pi * exp_x[i] / sum_exp**2
        dn_pi_dx_g = -exp_x_pi * exp_x[i] / sum_exp**2
        dn_dx = dn_dx.at[i, ll_x:ul_x].set(dn_g_dx_pi)
        dn_dx = dn_dx.at[ll:ul, i].set(dn_pi_dx_g)

        # Update dn_dx[i, i]
        dn_dx = dn_dx.at[i, i].set(exp_x[i] * (sum_exp - exp_x[i]) / sum_exp**2)

        # Update diagonal of dn_pi_dx_pi
        for j in range(ncwo):
            dn_pi_dx_pi = dn_pi_dx_pi.at[j, j].set(exp_x_pi[j] * (sum_exp - exp_x_pi[j]) / sum_exp**2)
        dn_dx = lax.dynamic_update_slice(dn_dx, dn_pi_dx_pi, (ll, ll_x))

        return n, dn_dx

    # Use a fori_loop to iterate over ndoc
    n, dn_dx = lax.fori_loop(0, ndoc, body_fun, (n, dn_dx))

    # print('jax_n',n)
    # print('jax_dn_dx',dn_dx) 
    # jax.debug.print("pynof_dn_dgamma -> {x}", x=dn_dx)
    # print('------------------------------------------------------------------------')
    # jax.debug.print("n -> {y}", y=n)
    return n,dn_dx

def alex_der_CJCKD5(n, ista, dn_dgamma, no1, ndoc, nalpha, nbeta, nv, nbf5, ndns, ncwo,p):

    # Interpair Electron correlation #
    Dcj12r = jnp.zeros((nbf5, nbf5, nv))
    Dck12r = jnp.zeros((nbf5, nbf5, nv))

    # Vectorized computation for interpair electron correlation
    def compute_interpair(k):
        Dcj12r_k = 2 * jnp.outer(dn_dgamma[:, k], n)
        Dck12r_k = jnp.outer(dn_dgamma[:, k], n)
        return Dcj12r_k, Dck12r_k

    Dcj12r, Dck12r = jax.vmap(compute_interpair)(jnp.arange(nv))
    Dcj12r = jnp.transpose(Dcj12r, (1, 2, 0))
    Dck12r = jnp.transpose(Dck12r, (1, 2, 0))

    # Intrapair Electron Correlation
    def compute_intrapair(i, state):
        Dcj12r, Dck12r = state  # Unpack the state
        l = i  # Loop index corresponds to l

        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1) * ncwo
        ul = ll + ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul]

        Dcj12r = Dcj12r.at[ldx, ll:ul, :nv].set(0)
        Dcj12r = Dcj12r.at[ll:ul, ldx, :nv].set(0)
        Dcj12r = Dcj12r.at[ll:ul, ll:ul, :nv].set(0)

        a = jnp.maximum(n_strong, 1e-15)
        b = jnp.maximum(n_weak, 1e-15)

        def update_Dck12r(k, Dck12r):
            dn_strong = dn_dgamma[ldx, k]
            dn_weak = dn_dgamma[ll:ul, k]
            Dck12r = Dck12r.at[ldx, ll:ul, k].set(0.5 * (1 / jnp.sqrt(a)) * dn_strong * jnp.sqrt(n_weak))
            Dck12r = Dck12r.at[ll:ul, ldx, k].set(0.5 * (1 / jnp.sqrt(b)) * dn_weak * jnp.sqrt(n_strong))
            Dck12r = Dck12r.at[ll:ul, ll:ul, k].set(-0.5 * jnp.outer((1 / jnp.sqrt(b)) * dn_weak, jnp.sqrt(n_weak)))
            return Dck12r

        Dck12r = lax.fori_loop(0, nv, update_Dck12r, Dck12r)
        return (Dcj12r, Dck12r)  # Return the updated state

    # Use lax.fori_loop to iterate over ndoc
    Dcj12r, Dck12r = lax.fori_loop(0, ndoc, compute_intrapair, (Dcj12r, Dck12r))
    # Dcj12r = Dcj12r.at[jnp.diag_indices(p.nbf5)].set(0)
    # Dck12r = Dck12r.at[jnp.diag_indices(p.nbf5)].set(0)
    # print('alex_',Dcj12r, Dck12r)
    return Dcj12r, Dck12r

def alex_der_CJCKD5_gamma(gamma,n, ista, dn_dgamma, no1, ndoc, nalpha, nbeta, nv, nbf5, ndns, ncwo):

    # Interpair Electron correlation #
    Dcj12r = jnp.zeros((nbf5, nbf5, nv))
    Dck12r = jnp.zeros((nbf5, nbf5, nv))

    # Vectorized computation for interpair electron correlation
    def compute_interpair(k):
        Dcj12r_k = 2 * jnp.outer(dn_dgamma[:, k], n)
        Dck12r_k = jnp.outer(dn_dgamma[:, k], n)
        return Dcj12r_k, Dck12r_k

    Dcj12r, Dck12r = jax.vmap(compute_interpair)(jnp.arange(nv))
    Dcj12r = jnp.transpose(Dcj12r, (1, 2, 0))
    Dck12r = jnp.transpose(Dck12r, (1, 2, 0))

    # Intrapair Electron Correlation
    def compute_intrapair(i, state):
        Dcj12r, Dck12r = state  # Unpack the state
        l = i  # Loop index corresponds to l

        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1) * ncwo
        ul = ll + ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul]

        Dcj12r = Dcj12r.at[ldx, ll:ul, :nv].set(0)
        Dcj12r = Dcj12r.at[ll:ul, ldx, :nv].set(0)
        Dcj12r = Dcj12r.at[ll:ul, ll:ul, :nv].set(0)

        a = jnp.maximum(n_strong, 1e-15)
        b = jnp.maximum(n_weak, 1e-15)

        def update_Dck12r(k, Dck12r):
            dn_strong = dn_dgamma[ldx, k]
            dn_weak = dn_dgamma[ll:ul, k]
            Dck12r = Dck12r.at[ldx, ll:ul, k].set(0.5 * (1 / jnp.sqrt(a)) * dn_strong * jnp.sqrt(n_weak))
            Dck12r = Dck12r.at[ll:ul, ldx, k].set(0.5 * (1 / jnp.sqrt(b)) * dn_weak * jnp.sqrt(n_strong))
            Dck12r = Dck12r.at[ll:ul, ll:ul, k].set(-0.5 * jnp.outer((1 / jnp.sqrt(b)) * dn_weak, jnp.sqrt(n_weak)))
            return Dck12r

        Dck12r = lax.fori_loop(0, nv, update_Dck12r, Dck12r)
        return (Dcj12r, Dck12r)  # Return the updated state

    # Use lax.fori_loop to iterate over ndoc
    Dcj12r, Dck12r = lax.fori_loop(0, ndoc, compute_intrapair, (Dcj12r, Dck12r))
    # jax.debug.print("Alex pynof Dcj12-> {x}", x=Dcj12r)
    return Dcj12r, Dck12r

def alex_der2_CJCKD5(n, dn_dgamma, d2n_d2gamma, no1, ndoc, nalpha, nbeta, nv, nbf5, ndns, ncwo):
    # Initialize second derivatives
    Dcj12r_second_deriv = jnp.zeros((nbf5, nbf5, nv, nv))
    Dck12r_second_deriv = jnp.zeros((nbf5, nbf5, nv, nv))

    # Compute second derivatives for interpair electron correlation
    def compute_interpair_second(k, l):
        d2Dcj12r_kl = 2 * jnp.outer(d2n_d2gamma[:, k, l], n)
        d2Dck12r_kl = jnp.outer(d2n_d2gamma[:, k, l], n)
        return d2Dcj12r_kl, d2Dck12r_kl

    Dcj12r_second_deriv, Dck12r_second_deriv = jax.vmap(lambda l: jax.vmap(lambda k: compute_interpair_second(k, l))(jnp.arange(nv)))(jnp.arange(nv))

    # Transpose to match desired shape (nbf5, nbf5, nv, nv)
    Dcj12r_second_deriv = jnp.transpose(Dcj12r_second_deriv, (2, 3, 0, 1))
    Dck12r_second_deriv = jnp.transpose(Dck12r_second_deriv, (2, 3, 0, 1))

    # Compute second derivatives for intrapair electron correlation
    def compute_intrapair_second(i, state):
        Dcj12r_second_deriv, Dck12r_second_deriv = state  # Unpack the state
        l = i  # Loop index corresponds to l

        ldx = no1 + l
        ll = no1 + ndns + (ndoc - l - 1) * ncwo
        ul = ll + ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul]

        Dcj12r_second_deriv = Dcj12r_second_deriv.at[ldx, ll:ul, :nv, :nv].set(0)
        Dcj12r_second_deriv = Dcj12r_second_deriv.at[ll:ul, ldx, :nv, :nv].set(0)
        Dcj12r_second_deriv = Dcj12r_second_deriv.at[ll:ul, ll:ul, :nv, :nv].set(0)

        a = jnp.maximum(n_strong, 1e-15)
        b = jnp.maximum(n_weak, 1e-15)

        def update_Dck12r_second(k, l, Dck12r_second_deriv):
            dn_strong = dn_dgamma[ldx, k]
            dn_weak = dn_dgamma[ll:ul, k]
            d2n_strong = d2n_d2gamma[ldx, k, l]
            d2n_weak = d2n_d2gamma[ll:ul, k, l]

            sqrt_a = jnp.sqrt(a)
            sqrt_b = jnp.sqrt(b)
            inv_sqrt_a = 1 / sqrt_a
            inv_sqrt_b = 1 / sqrt_b

            Dck12r_second_deriv = Dck12r_second_deriv.at[ldx, ll:ul, k, l].set(
                0.5 * inv_sqrt_a * (d2n_strong * sqrt_b + dn_strong * d2n_weak)
            )
            Dck12r_second_deriv = Dck12r_second_deriv.at[ll:ul, ldx, k, l].set(
                0.5 * inv_sqrt_b * (d2n_weak * sqrt_a + dn_weak * d2n_strong)
            )
            Dck12r_second_deriv = Dck12r_second_deriv.at[ll:ul, ll:ul, k, l].set(
                -0.5 * jnp.outer(inv_sqrt_b * d2n_weak, sqrt_b)
            )
            return Dck12r_second_deriv

        Dck12r_second_deriv = lax.fori_loop(
            0, nv, 
            lambda k, Dck12r_second_deriv: lax.fori_loop(
                0, nv, lambda l, Dck12r_second_deriv: update_Dck12r_second(k, l, Dck12r_second_deriv),
                Dck12r_second_deriv
            ),
            Dck12r_second_deriv
        )

        return Dcj12r_second_deriv, Dck12r_second_deriv  # Return updated state

    # Iterate over ndoc to compute second derivatives
    Dcj12r_second_deriv, Dck12r_second_deriv = lax.fori_loop(0, ndoc, compute_intrapair_second, (Dcj12r_second_deriv, Dck12r_second_deriv))

    # Set diagonal elements to zero
    Dcj12r_second_deriv = Dcj12r_second_deriv.at[jnp.diag_indices(nbf5)].set(0)
    Dck12r_second_deriv = Dck12r_second_deriv.at[jnp.diag_indices(nbf5)].set(0)

    return Dcj12r_second_deriv, Dck12r_second_deriv


def alex_der_CJCKD5_second_derivatives(n, dn_dgamma, d2n_d2gamma, no1, ndoc, nv, nbf5, ndns, ncwo):
    """
    Compute the second derivatives of Dcj12r and Dck12r with respect to gamma using JAX.
    
    Args:
        n: Occupation numbers (array of shape (nv,)).
        dn_dgamma: First derivative of n with respect to gamma (array of shape (nv, nv)).
        d2n_d2gamma: Second derivative of n with respect to gamma (array of shape (nv, nv, nv)).
        no1, ndoc, nalpha, nbeta, nv, nbf5, ndns, ncwo: Parameters for the function.
        
    Returns:
        Dcj12r_second_deriv: Second derivative of Dcj12r (array of shape (nbf5, nbf5, nv, nv)).
        Dck12r_second_deriv: Second derivative of Dck12r (array of shape (nbf5, nbf5, nv, nv)).
    """
    # Initialize second derivatives
    Dcj12r_second_deriv = jnp.zeros((nbf5, nbf5, nv, nv))
    Dck12r_second_deriv = jnp.zeros((nbf5, nbf5, nv, nv))
    
    # Interpair Electron Correlation
    for k in range(nv):
        for l in range(nv):
            # Second derivative of Dcj12r
            Dcj12r_second_deriv = Dcj12r_second_deriv.at[:, :, k, l].set(
                2 * (
                    d2n_d2gamma[:, k, l] * n +
                    dn_dgamma[:, k] * dn_dgamma[:, l] +
                    dn_dgamma[:, l] * dn_dgamma[:, k] +
                    dn_dgamma[:, k] * d2n_d2gamma[:, l]
                )
            )
            # Second derivative of Dck12r
            Dck12r_second_deriv = Dck12r_second_deriv.at[:, :, k, l].set(
                d2n_d2gamma[:, k, l] * n +
                dn_dgamma[:, k] * dn_dgamma[:, l] +
                dn_dgamma[:, l] * dn_dgamma[:, k] +
                dn_dgamma[:, k] * d2n_d2gamma[:, l]
            )

    # Intrapair Electron Correlation
    for l in range(ndoc):
        ldx = no1 + l
        ll = no1 + ndns + (ndoc - l - 1) * ncwo
        ul = ll + ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul]

        a = jnp.maximum(n_strong, 1e-15)
        b = jnp.where(n_weak < 1e-15, 1e-15, n_weak)

        for k in range(nv):
            for m in range(nv):
                dn_strong_k = dn_dgamma[ldx, k]
                dn_weak_k = dn_dgamma[ll:ul, k]
                dn_strong_m = dn_dgamma[ldx, m]
                dn_weak_m = dn_dgamma[ll:ul, m]

                d2n_strong_km = d2n_d2gamma[ldx, k, m]
                d2n_weak_km = d2n_d2gamma[ll:ul, k, m]

                # Second derivative of Dck12r
                Dck12r_second_deriv = Dck12r_second_deriv.at[ldx, ll:ul, k, m].set(
                    0.5 * (1 / jnp.sqrt(a)) * d2n_strong_km * jnp.sqrt(n_weak) +
                    0.5 * (1 / jnp.sqrt(a)) * dn_strong_k * (1 / (2 * jnp.sqrt(n_weak))) * dn_weak_m
                )
                Dck12r_second_deriv = Dck12r_second_deriv.at[ll:ul, ldx, k, m].set(
                    0.5 * (1 / jnp.sqrt(b)) * d2n_weak_km * jnp.sqrt(n_strong) +
                    0.5 * (1 / jnp.sqrt(b)) * dn_weak_k * (1 / (2 * jnp.sqrt(n_strong))) * dn_strong_m
                )
                Dck12r_second_deriv = Dck12r_second_deriv.at[ll:ul, ll:ul, k, m].set(
                    -0.5 * (1 / jnp.sqrt(b) * d2n_weak_km * jnp.sqrt(n_weak) +
                    1 / jnp.sqrt(b) * dn_weak_k * (1 / (2 * jnp.sqrt(n_weak))) * dn_weak_m
                ))

    return Dcj12r_second_deriv, Dck12r_second_deriv

def alex_CJCKD5(n, no1, ndoc, nsoc, nbeta, nalpha, ndns, ncwo, MSpin):
    """
    PNOF5 coefficients C^J and C^K that multiply J and K integrals.

    E = 2\sum_p n_p H_p + \sum_{pq} C^J_{pq} J_{qp} - \sum_{pq} C^K_{pq} K_{qp}

    C^J_{pq} = \begin{cases}
                 2 n_p n_q & \text{if } p \in \Omega_g, q \in \Omega_f, g \neq f \\
                 0         & \text{otherwise}
               \end{cases}

    C^K_{pq} = \begin{cases}
                 n_p n_q        & \text{if } p \in \Omega_g, q \in \Omega_f, g \neq f \\
                 \sqrt{n_p n_q} & \text{if } p, q \in \Omega_g \text{ and } ((p \leq F, q > F) \text{ or } (p > F, q \leq F)) \\
                -\sqrt{n_p n_q} & \text{if } p \neq q \in \Omega_g \text{ and } p, q > F \\
                 0              & \text{otherwise}
               \end{cases}
    """

    # Interpair Electron correlation
    cj12 = 2 * jnp.outer(n, n)
    ck12 = jnp.outer(n, n)

    # Intrapair Electron Correlation
    if MSpin == 0 and nsoc > 1:
        ck12 = ck12.at[nbeta:nalpha, nbeta:nalpha].set(2 * jnp.outer(n[nbeta:nalpha], n[nbeta:nalpha]))

    # Loop over strongly occupied orbitals
    def body_fun(l, cj12_ck12):
        cj12, ck12 = cj12_ck12

        ldx = no1 + l
        ll = no1 + ndns + (ndoc - l - 1) * ncwo
        ul = ll + ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul]

        # Update cj12
        cj12 = cj12.at[ldx, ll:ul].set(0)
        cj12 = cj12.at[ll:ul, ldx].set(0)
        cj12 = cj12.at[ll:ul, ll:ul].set(0)

        # Update ck12
        ck12 = ck12.at[ldx, ll:ul].set(jnp.sqrt(n_strong * n_weak))
        ck12 = ck12.at[ll:ul, ldx].set(jnp.sqrt(n_strong * n_weak))
        ck12 = ck12.at[ll:ul, ll:ul].set(-jnp.sqrt(jnp.outer(n_weak, n_weak)))

        return cj12, ck12

    # Use lax.fori_loop for the loop
    cj12, ck12 = lax.fori_loop(0, ndoc, body_fun, (cj12, ck12))
    cj12 = cj12.at[jnp.diag_indices_from(cj12)].set(0)
    ck12 = ck12.at[jnp.diag_indices_from(cj12)].set(0)
    # jax.debug.print('alex_ck12_j{x}',x=ck12)
    return  cj12,ck12

def alex_calcoccg_finite_diff(gamma, J_MO, K_MO, H_core, p, h=1e-5):
    """
    Compute the gradient using finite differences.

    Args:
        gamma: Input variable (array of shape (p.nv,)).
        J_MO, K_MO, H_core: Matrices involved in the computation.
        p: Object containing parameters (nv, nbf5, etc.).
        h: Small perturbation for finite differences (default: 1e-5).

    Returns:
        grad: Approximated gradient (array of shape (p.nv,)).
    """
    nv = p.nv
    grad = np.zeros(nv)

    # Base computation
    n_base, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    dn_dgamma_fd = np.zeros((nv, nv))
    for k in range(nv):
        gamma_plus = gamma.copy()
        gamma_minus = gamma.copy()
        gamma_plus[k] += h
        gamma_minus[k] -= h

        n_plus, _ = ocupacion(gamma_plus, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
        n_minus, _ = ocupacion(gamma_minus, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)

        dn_dgamma_fd[:, k] = (n_plus - n_minus) / (2 * h)
    Dcj12r_base, Dck12r_base = der_PNOFi_selector(n_base,dn_dgamma_fd, p)

    # Compute grad using finite differences
    if p.MSpin == 0:
        # Contribution from dn_dgamma (2H + J)
        grad += np.einsum('ik,i->k', dn_dgamma_fd, 2 * H_core + np.diagonal(J_MO), optimize=True)

        # Contribution from Dcj12r and Dck12r
        diag = np.diag_indices(p.nbf5)
        Dcj12r_base[diag] = 0
        grad += 2 * np.einsum('ijk,ji->k', Dcj12r_base, J_MO, optimize=True)

        Dck12r_base[diag] = 0
        grad -= 2 * np.einsum('ijk,ji->k', Dck12r_base, K_MO, optimize=True)

    return grad


def compute_energy(gamma, J_MO, K_MO, H_core, p):
    """
    Compute the energy E as a function of gamma.

    Args:
        gamma: Input variable (array of shape (p.nv,)).
        J_MO, K_MO, H_core: Matrices involved in the computation.
        p: Object containing parameters (nbeta, nalpha, nbf5, etc.).

    Returns:
        E: Scalar energy value.
    """
    # Compute occupation numbers and their derivatives
    n, _ = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    cj12, ck12 = der_PNOFi_selector(n, np.zeros((p.nv, p.nv)), p)

    # Compute energy contributions
    E = 0.0

    # Contribution from n and H_core
    E = E + 2 * np.einsum('i,i', n, H_core, optimize=True)

    # Contribution from n and J_MO (beta and alpha parts)
    E = E + np.einsum('i,i', n[:p.nbeta], np.diagonal(J_MO)[:p.nbeta], optimize=True)
    E = E + np.einsum('i,i', n[p.nalpha:p.nbf5], np.diagonal(J_MO)[p.nalpha:p.nbf5], optimize=True)

    # Contribution from cj12 and J_MO
    np.fill_diagonal(cj12, 0)  # Remove diagonal
    E = E + np.einsum('ijk,ji->', cj12, J_MO, optimize=True)

    # Contribution from ck12 and K_MO
    np.fill_diagonal(ck12, 0)  # Remove diagonal
    E = E - np.einsum('ijk,ji->', ck12, K_MO, optimize=True)

    return E

def compute_gradient_finite_diff(gamma, J_MO, K_MO, H_core, p, h=1e-5):
    """
    Compute the gradient of E with respect to gamma using finite differences.

    Args:
        gamma: Input variable (array of shape (p.nv,)).
        J_MO, K_MO, H_core: Matrices involved in the computation.
        p: Object containing parameters (nbeta, nalpha, nbf5, etc.).
        h: Small perturbation for finite differences (default: 1e-5).

    Returns:
        grad: Approximated gradient (array of shape (p.nv,)).
    """
    nv = gamma.shape[0]
    grad = np.zeros(nv)

    # Base energy
    E_base = compute_energy(gamma, J_MO, K_MO, H_core, p)

    # Compute gradient using finite differences
    for i in range(nv):
        # Perturb gamma[i] by +h
        gamma_plus = gamma.copy()
        gamma_plus[i] += h
        E_plus = compute_energy(gamma_plus, J_MO, K_MO, H_core, p)

        # Perturb gamma[i] by -h
        gamma_minus = gamma.copy()
        gamma_minus[i] -= h
        E_minus = compute_energy(gamma_minus, J_MO, K_MO, H_core, p)

        # Central difference formula
        grad[i] = (E_plus - E_minus) / (2 * h)

    return grad

def alex_calchess_finite_diff(gamma, J_MO, K_MO, H_core, p, h=1e-5):
    """
    Compute the Hessian of E with respect to gamma using finite differences,
    including second derivatives of n, cj12, and ck12.

    Args:
        gamma: Input variable (array of shape (p.nv,)).
        J_MO, K_MO, H_core: Matrices involved in the computation.
        p: Object containing parameters (nbeta, nalpha, nbf5, etc.).
        h: Small perturbation for finite differences (default: 1e-5).

    Returns:
        hessian: Approximated Hessian matrix (array of shape (p.nv, p.nv)).
    """
    nv = gamma.shape[0]
    hessian = np.zeros((nv, nv))

    # Compute second derivatives of n, cj12, and ck12 using finite differences
    d2n_d2gamma = np.zeros((nv, nv, p.nv))  # Shape: (nv, nv, nv)
    d2cj12_d2gamma = np.zeros((nv, nv, p.nbf5, p.nbf5, p.nv))  # Shape: (nv, nv, nbf5, nbf5, nv)
    d2ck12_d2gamma = np.zeros((nv, nv, p.nbf5, p.nbf5, p.nv))  # Shape: (nv, nv, nbf5, nbf5, nv)

    for i in range(nv):
        for j in range(nv):
            # Perturb gamma[i] and gamma[j] by +h and -h
            gamma_plus_plus = gamma.copy()
            gamma_plus_plus[i] += h
            gamma_plus_plus[j] += h

            gamma_plus_minus = gamma.copy()
            gamma_plus_minus[i] += h
            gamma_plus_minus[j] -= h

            gamma_minus_plus = gamma.copy()
            gamma_minus_plus[i] -= h
            gamma_minus_plus[j] += h

            gamma_minus_minus = gamma.copy()
            gamma_minus_minus[i] -= h
            gamma_minus_minus[j] -= h

            # Compute n, cj12, and ck12 at perturbed points
            n_plus_plus, dn_dgamma = ocupacion(gamma_plus_plus, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
            cj12_plus_plus, ck12_plus_plus = der_PNOFi_selector(n_plus_plus, dn_dgamma, p)

            n_plus_minus, _ = ocupacion(gamma_plus_minus, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
            cj12_plus_minus, ck12_plus_minus = der_PNOFi_selector(n_plus_minus, dn_dgamma, p)

            n_minus_plus, _ = ocupacion(gamma_minus_plus, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
            cj12_minus_plus, ck12_minus_plus = der_PNOFi_selector(n_minus_plus, dn_dgamma, p)

            n_minus_minus, _ = ocupacion(gamma_minus_minus, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
            cj12_minus_minus, ck12_minus_minus = der_PNOFi_selector(n_minus_minus, dn_dgamma, p)

            # Central difference formula for second derivatives
            d2n_d2gamma[i, j] = (n_plus_plus - n_plus_minus - n_minus_plus + n_minus_minus) / (4 * h**2)
            d2cj12_d2gamma[i, j] = (cj12_plus_plus - cj12_minus_minus) / (2 * h)
            d2ck12_d2gamma[i, j] = (ck12_plus_plus -  ck12_minus_minus) / (2 * h)

    # Compute Hessian contributions
    if p.MSpin == 0:
        # Contribution from second derivatives of n (2H + J)
        for i in range(nv):
            for j in range(nv):
                hessian[i, j] += np.einsum('k,k->', d2n_d2gamma[i, j], 2 * H_core + np.diagonal(J_MO), optimize=True)
                hessian[i, j] += 2 * np.einsum('klm,lm->', d2cj12_d2gamma[i, j], J_MO, optimize=True)
                hessian[i, j] -= 2 * np.einsum('klm,lm->', d2ck12_d2gamma[i, j], K_MO, optimize=True)
        
    return hessian


def compute_hessian_finite_diff(gamma, J_MO, K_MO, H_core, p, h=1e-5):
    """
    Compute the Hessian of E with respect to gamma using finite differences.

    Args:
        gamma: Input variable (array of shape (p.nv,)).
        J_MO, K_MO, H_core: Matrices involved in the computation.
        p: Object containing parameters (nbeta, nalpha, nbf5, etc.).
        h: Small perturbation for finite differences (default: 1e-5).

    Returns:
        hessian: Approximated Hessian matrix (array of shape (p.nv, p.nv)).
    """
    nv = gamma.shape[0]
    hessian = np.zeros((nv, nv))

    # Compute Hessian using finite differences
    for i in range(nv):
        for j in range(nv):
            # Perturb gamma[i] and gamma[j] by +h and -h
            gamma_plus_plus = gamma.copy()
            gamma_plus_plus[i] += h
            gamma_plus_plus[j] += h

            gamma_plus_minus = gamma.copy()
            gamma_plus_minus[i] += h
            gamma_plus_minus[j] -= h

            gamma_minus_plus = gamma.copy()
            gamma_minus_plus[i] -= h
            gamma_minus_plus[j] += h

            gamma_minus_minus = gamma.copy()
            gamma_minus_minus[i] -= h
            gamma_minus_minus[j] -= h

            # Compute energies for perturbed gamma
            E_plus_plus = compute_energy(gamma_plus_plus, J_MO, K_MO, H_core, p)
            E_plus_minus = compute_energy(gamma_plus_minus, J_MO, K_MO, H_core, p)
            E_minus_plus = compute_energy(gamma_minus_plus, J_MO, K_MO, H_core, p)
            E_minus_minus = compute_energy(gamma_minus_minus, J_MO, K_MO, H_core, p)

            # Central difference formula for Hessian
            hessian[i, j] = (E_plus_plus - E_plus_minus - E_minus_plus + E_minus_minus) / (4 * h**2)

    return hessian


def Alex_ocupacion_softmax(x, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin):
    """Transform gammas to n according to the softmax 
    parameterization of the occupation numbers"""
    n = np.zeros(nbf5, dtype=np.float64)
    dn_dx = np.zeros((nbf5, nv), dtype=np.float64)

    # Initialize n for HighSpin or non-HighSpin cases
    if not HighSpin:
        n[no1 + ndoc : no1 + ndns] = 0.5  # (no1+ndoc, no1+ndns]
    else:
        n[no1 + ndoc : no1 + ndns] = 1.0  # (no1+ndoc, no1+ndns]

    exp_x = np.exp(x)

    for i in range(ndoc):
        ll = no1 + ndns + (ndoc - i - 1) * ncwo
        ul = ll + ncwo
        n_pi = n[ll:ul]

        ll_x = ll - ndns + ndoc - no1
        ul_x = ll_x + ncwo
        exp_x_pi = exp_x[ll_x:ul_x]

        sum_exp = exp_x[i] + np.sum(exp_x_pi)

        # Compute n[i] and n_pi
        n[i] = exp_x[i] / sum_exp
        n_pi[:] = exp_x_pi / sum_exp

        # Compute dn_pi_dx_pi
        for j in range(ncwo):
            for k in range(ncwo):
                dn_dx[ll + j, ll_x + k] = -exp_x_pi[j] * exp_x_pi[k] / sum_exp**2

        # Compute dn_g_dx_pi and dn_pi_dx_g
        for j in range(ncwo):
            dn_dx[i, ll_x + j] = -exp_x_pi[j] * exp_x[i] / sum_exp**2
            dn_dx[ll + j, i] = -exp_x_pi[j] * exp_x[i] / sum_exp**2

        # Compute dn_dx[i, i]
        dn_dx[i, i] = exp_x[i] * (sum_exp - exp_x[i]) / sum_exp**2

        # Compute diagonal elements of dn_pi_dx_pi
        for j in range(ncwo):
            dn_dx[ll + j, ll_x + j] = exp_x_pi[j] * (sum_exp - exp_x_pi[j]) / sum_exp**2

    return n, dn_dx