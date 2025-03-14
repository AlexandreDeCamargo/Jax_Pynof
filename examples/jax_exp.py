import jax


# def f(x): return jax.numpy.sin(jax.numpy.cos(x))

f = lambda x : jax.numpy.sin(jax.numpy.cos(x))
print(jax.make_jaxpr(f)(3.0))
print(jax.make_jaxpr(jax.grad(f))(3.0))