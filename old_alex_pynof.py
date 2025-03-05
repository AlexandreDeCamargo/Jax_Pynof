import jax 
import jax.numpy as jnp 
from pynof.pnof import PNOFi_selector, ocupacion,der_PNOFi_selector

from jax import lax

jax.config.update("jax_enable_x64", True)

def calce(n, cj12, ck12, J_MO, K_MO, H_core, p):
    E = 0.0
    n = jnp.asarray(n)
    cj12 = jnp.asarray(cj12)
    ck12 = jnp.asarray(ck12)
    # print('n',n)

    if p.MSpin == 0:
        E += 2 * jnp.einsum('i,i', n, H_core, optimize=True)
        E += jnp.einsum('i,i', n[:p.nbeta], jnp.diagonal(J_MO)[:p.nbeta], optimize=True)
        E += jnp.einsum('i,i', n[p.nalpha:p.nbf5], jnp.diagonal(J_MO)[p.nalpha:p.nbf5], optimize=True)
        
        cj12 = cj12.at[jnp.diag_indices_from(cj12)].set(0)
        E += jnp.einsum('ij,ji->', cj12, J_MO, optimize=True)
        
        ck12 = ck12.at[jnp.diag_indices_from(ck12)].set(0)
        E -= jnp.einsum('ij,ji->', ck12, K_MO, optimize=True)
    
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

    return E

def calcocce(gamma,J_MO,K_MO,H_core,p):

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
    cj12,ck12 = PNOFi_selector(n,p)

    n = jnp.asarray(n)
    cj12 = jnp.asarray(cj12)
    ck12 = jnp.asarray(ck12)

    # print('occ numbers',n)
    E = calce(n,cj12,ck12,J_MO,K_MO,H_core,p)
    
    
    return E

def ocupation_trigonometric(gamma, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin):
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

def calcoccg_jax(gamma, J_MO, K_MO, H_core, p):
    n,dn_dgamma = ocupation_trigonometric(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin)
   
    # cj12, ck12 = der_PNOFi_selector(n, dn_dgamma, p)
    cj12,ck12 = PNOFi_selector(n,p)

    n = jnp.asarray(n)
    cj12 = jnp.asarray(cj12)
    ck12 = jnp.asarray(ck12)

    # Compute energy
    E = calce(n, cj12, ck12, J_MO, K_MO, H_core, p)
    
    return E  # Return energy so JAX can differentiate it


def calcoccg(gamma, J_MO, K_MO, H_core, p):
    grad = jnp.zeros((p.nv))
    # n,dn_dgamma = ocupation_trigonometric(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin)
    n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    Dcj12r, Dck12r = der_PNOFi_selector(n, dn_dgamma, p)
    n = jnp.asarray(n)
    dn_dgamma = jnp.array(dn_dgamma)
    Dcj12r = jnp.array(Dcj12r)
    Dck12r = jnp.array(Dck12r)
    
    if p.MSpin == 0:
        grad += jnp.einsum('ik,i->k', dn_dgamma, 2 * H_core + jnp.diagonal(J_MO), optimize=True)
        
        Dcj12r = Dcj12r.at[jnp.diag_indices(p.nbf5)].set(0)
        grad += 2 * jnp.einsum('ijk,ji->k', Dcj12r, J_MO, optimize=True)
        
        Dck12r = Dck12r.at[jnp.diag_indices(p.nbf5)].set(0)
        grad -= 2 * jnp.einsum('ijk,ji->k', Dck12r, K_MO, optimize=True)
        
    else:
        grad += jnp.einsum(
            'ik,i->k',
            dn_dgamma[p.no1:p.nbeta, :p.nv],
            2 * H_core[p.no1:p.nbeta] + jnp.diagonal(J_MO)[p.no1:p.nbeta],
            optimize=True
        )
        grad += jnp.einsum(
            'ik,i->k',
            dn_dgamma[p.nalpha:p.nbf5, :p.nv],
            2 * H_core[p.nalpha:p.nbf5] + jnp.diagonal(J_MO)[p.nalpha:p.nbf5],
            optimize=True
        )
        
        Dcj12r = Dcj12r.at[jnp.diag_indices(p.nbf5)].set(0)
        grad += 2 * jnp.einsum('ijk,ji->k', Dcj12r[p.no1:p.nbeta, :p.nbeta, :p.nv], J_MO[:p.nbeta, p.no1:p.nbeta], optimize=True)
        grad += 2 * jnp.einsum('ijk,ji->k', Dcj12r[p.no1:p.nbeta, p.nalpha:p.nbf5, :p.nv], J_MO[p.nalpha:p.nbf5, p.no1:p.nbeta], optimize=True)
        grad += 2 * jnp.einsum('ijk,ji->k', Dcj12r[p.nalpha:p.nbf5, :p.nbeta, :p.nv], J_MO[:p.nbeta, p.nalpha:p.nbf5], optimize=True)
        grad += 2 * jnp.einsum('ijk,ji->k', Dcj12r[p.nalpha:p.nbf5, p.nalpha:p.nbf5, :p.nv], J_MO[p.nalpha:p.nbf5, p.nalpha:p.nbf5], optimize=True)
        
        Dck12r = Dck12r.at[jnp.diag_indices(p.nbf5)].set(0)
        grad -= 2 * jnp.einsum('ijk,ji->k', Dck12r[p.no1:p.nbeta, :p.nbeta, :p.nv], K_MO[:p.nbeta, p.no1:p.nbeta], optimize=True)
        grad -= 2 * jnp.einsum('ijk,ji->k', Dck12r[p.no1:p.nbeta, p.nalpha:p.nbf5, :p.nv], K_MO[p.nalpha:p.nbf5, p.no1:p.nbeta], optimize=True)
        grad -= 2 * jnp.einsum('ijk,ji->k', Dck12r[p.nalpha:p.nbf5, :p.nbeta, :p.nv], K_MO[:p.nbeta, p.nalpha:p.nbf5], optimize=True)
        grad -= 2 * jnp.einsum('ijk,ji->k', Dck12r[p.nalpha:p.nbf5, p.nalpha:p.nbf5, :p.nv], K_MO[p.nalpha:p.nbf5, p.nalpha:p.nbf5], optimize=True)
        
        grad += 2 * jnp.einsum('jk,ij->k', dn_dgamma[p.no1:p.nbeta, :p.nv], J_MO[p.nbeta:p.nalpha, p.no1:p.nbeta], optimize=True)
        grad += 2 * jnp.einsum('jk,ij->k', dn_dgamma[p.nalpha:p.nbf5, :p.nv], J_MO[p.nbeta:p.nalpha, p.nalpha:p.nbf5], optimize=True)
        
        grad -= jnp.einsum('jk,ij->k', dn_dgamma[p.no1:p.nbeta, :p.nv], K_MO[p.nbeta:p.nalpha, p.no1:p.nbeta], optimize=True)
        grad -= jnp.einsum('jk,ij->k', dn_dgamma[p.nalpha:p.nbf5, :p.nv], K_MO[p.nbeta:p.nalpha, p.nalpha:p.nbf5], optimize=True)
    
    return grad


def jax_ocupacion_softmax(x, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin):
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

    return n, dn_dx