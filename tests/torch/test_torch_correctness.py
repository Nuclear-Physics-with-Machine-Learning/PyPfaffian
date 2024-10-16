import pytest

import torch

def test_pfaffian_correctness(N, seed, dtype):

    from py_pfaffian.torch import pfaffian

    # For JAX, create a random matrix then Anti-symmetrize it:
    torch.manual_seed(int(seed))

    matrix = torch.rand(size=(N, N))

    # Antisymmetrization:
    matrix = matrix - matrix.T

    pf_ltl = pfaffian(matrix, method="LTL")

    print(matrix)

    print("LTL: ", pf_ltl)

    if N <=4 :
        pf_direct = pfaffian(matrix, method="direct")
        print("Direct, ", pf_direct)
        assert torch.allclose(pf_direct, pf_ltl) 
    if N < 16:
        pf_rec = pfaffian(matrix, method="recursive")

        print("Recursive: ", pf_rec)

        assert torch.allclose(pf_ltl, pf_rec)

    # Last check, need to ensure the pfaffian squared matches the determinant:

    det = torch.linalg.det(matrix)

    assert torch.allclose(pf_ltl**2, det)



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


