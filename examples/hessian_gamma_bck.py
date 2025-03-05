def H_gamma(gamma, J_MO, K_MO, H_core, p, dndg):
    """
    Compute the Hessian of the energy with respect to gamma.
    """
    grad = jnp.zeros((p.nv))

    dn_dgamma = jnp.array(dndg)
    
    d2n_d2gamma = jax.jacobian(dn_dgamma_,argnums=0)
    d2n_d2gamma = d2n_d2gamma(gamma)[0]
    
    # dn_dgamma = jnp.array(dndg)

    jax.debug.print("🤯 dn_dgamma{y} 🤯", y=dn_dgamma)
    jax.debug.print("🤯 n{x} 🤯", x=n)
    # print('gamma',gamma)
    # Dcj12r, Dck12r = alex_der_CJCKD5(n, p.ista, dn_dgamma, p.no1, p.ndoc, 
    #                                  p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo,p)
    # Dcj12r, Dck12r = jnp.asarray(Dcj12r), jnp.asarray(Dck12r)

    Dcj12r_dgamma,Dck12r_dgamma = alex_der2_CJCKD5(n, dn_dgamma, d2n_d2gamma, p.no1, 
                                                   p.ndoc, p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo)

    grad += jnp.einsum('ik,i->k',dn_dgamma,2*H_core+jnp.diagonal(J_MO),optimize=True)

    # 2 dCJ_dgamma J_MO

    Dcj12r_dgamma =  Dcj12r_dgamma.at[jnp.diag_indices(p.nbf5)].set(0)
    # Dcj12r = Dcj12r.at[jnp.diag_indices(p.nbf5)].set(0)
    grad += 2 * jnp.einsum('ijk,ji->k', Dcj12r_dgamma, J_MO, optimize=True)
    
    Dck12r_dgamma = Dck12r_dgamma.at[jnp.diag_indices(p.nbf5)].set(0)
    # Dck12r = Dck12r.at[jnp.diag_indices(p.nbf5)].set(0)
    grad -= 2 * jnp.einsum('ijk,ji->k', Dck12r_dgamma, K_MO, optimize=True)
    
    return grad


hh_gamma = H_gamma(gamma, J_MO, K_MO, H_core, p,dn_dgamma)
print('hessian_gamma',hh_gamma)
assert 0 
