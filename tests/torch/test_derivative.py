import pytest

import torch


# def test_pfaffian_jvp(N, seed, dtype):

#     from py_pfaffian.torch import Pfaffian

#     torch.manual_seed(int(seed))

#     matrix = torch.rand(size=(N, N))

#     matrix.requires_grad_(True)

#     # Antisymmetrization:
#     matrix = matrix - matrix.T

#     print(matrix)

#     tangent = torch.zeros_like(matrix)
#     tangent[0,3] += 1.0

#     tangent = tangent - tangent.T


#     value, gradients = Pfaffian.apply, ), (matrix,), (tangent,))
#     print("Gradients: ", gradients)

#     kick=1-5
#     numerical_pf_dot = (1/kick) *( Pfaffian.apply(matrix + 0.5*kick * tangent) - Pfaffian.apply(matrix - 0.5*kick * tangent))

#     print("Numerical gradients: ", numerical_pf_dot)

#     assert torch.allclose(numerical_pf_dot, pf_ltl.grad)

#     # print(grad_value)




def test_pfaffian_vjp(N, seed, dtype):

    from py_pfaffian.jax import pfaffian

    # For JAX, create a random matrix then Anti-symmetrize it:

    torch.manual_seed(int(seed))

    matrix = torch.rand(size=(N, N))

    matrix.requires_grad_(True)

    # Antisymmetrization:
    matrix = matrix - matrix.T

    pf = Pfaffian.apply()

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


