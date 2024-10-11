import pytest

import jax
from jax import config; config.update("jax_enable_x64", True)


def test_pfaffian_jvp(N, seed, dtype):

    from py_pfaffian.jax import pfaffian

    # For JAX, create a random matrix then Anti-symmetrize it:


    key = jax.random.PRNGKey(int(seed))
    matrix = jax.random.uniform(key, (N, N))

    # Antisymmetrization:
    matrix = matrix - matrix.T

    # pf_ltl = pfaffian(matrix, method="LTL")


    tangent = jax.numpy.zeros_like(matrix)
    tangent = tangent.at[0,3].add(1.0)

    # For a pfaffian, can't move one side without moving the other:
    tangent = tangent - tangent.T

    pf, pf_dot =  jax.jvp(pfaffian, (matrix,), (tangent,) )

    # pf_ltl, grad_value = pf_grad_fn(matrix)

    kick=1-5
    numerical_pf_dot = (1/kick) *( pfaffian(matrix + 0.5*kick * tangent) - pfaffian(matrix - 0.5*kick * tangent))


    assert jax.numpy.allclose(numerical_pf_dot, pf_dot)

    # print(grad_value)




def test_pfaffian_vjp(N, seed, dtype):

    from py_pfaffian.jax import pfaffian

    # For JAX, create a random matrix then Anti-symmetrize it:


    key = jax.random.PRNGKey(int(seed))
    matrix = jax.random.uniform(key, (N, N))

    # Antisymmetrization:
    matrix = matrix - matrix.T

    # pf_ltl = pfaffian(matrix, method="LTL")


    tangent = jax.numpy.zeros_like(matrix)
    tangent = tangent.at[0,3].add(1.0)

    # For a pfaffian, can't move one side without moving the other:
    tangent = tangent - tangent.T

    pf, vjp_fn = jax.vjp(pfaffian, matrix)


    vjp = vjp_fn(1.0)








# def test_log_pfaffian_correctness(N, seed, dtype):

#     from py_pfaffian.jax import log_pfaffian

#     # For JAX, create a random matrix then Anti-symmetrize it:


#     key = jax.random.PRNGKey(int(seed))
#     matrix = jax.random.uniform(key, (N, N))

#     # Antisymmetrization:
#     matrix = matrix - matrix.T

#     sign_LTL, log_pf_ltl = log_pfaffian(matrix, method="LTL")

#     print(matrix)

#     print("LTL: ", log_pf_ltl)

#     if N <=4 :
#         sign_direct, log_pf_direct = log_pfaffian(matrix, method="direct")
#         print("Direct, ", log_pf_ltl)
#         print("Sign direct: ", sign_direct)
#         print("Sign LTL: ", sign_LTL)
#         assert jax.numpy.allclose(sign_LTL, sign_direct) 
#         assert jax.numpy.allclose(log_pf_ltl, log_pf_direct) 
   

#     # Last check, need to ensure the pfaffian squared matches the determinant:

#     sign, logdet = jax.numpy.linalg.slogdet(matrix)


#     print(logdet)
#     print(log_pf_ltl)

#     # We don't check the sign, since it can be either way.
#     assert jax.numpy.allclose(log_pf_ltl*2, logdet)


