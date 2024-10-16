import pytest

import numpy as np

def test_pfaffian_correctness(N, seed, dtype):

    import torch
    import jax.numpy as jnp

    from py_pfaffian.torch import pfaffian as pf_torch
    from py_pfaffian.jax import pfaffian as pf_jax



    matrix = np.random.uniform(size=(N, N))

    # Antisymmetrization:
    matrix = matrix - matrix.T


    if N <=4 :
        pf_direct_J = pf_jax(matrix, method="direct")
        pf_direct_T = pf_torch(torch.as_tensor(matrix), method="direct")
        print("Direct J, ", pf_direct_J)
        print("Direct T, ", pf_direct_T)
        assert np.allclose(pf_direct_J, pf_direct_T.cpu()) 
    if N < 16:
        pf_rec_T = pf_torch(torch.as_tensor(matrix), method="recursive")
        pf_rec_J = pf_jax(matrix, method="recursive")

        print("Recursive J: ", pf_rec_J)
        print("Recursive T: ", pf_rec_T)

        assert np.allclose(pf_rec_J, pf_rec_T.cpu())

    # CHeck the decompositions:
    pf_ltl_torch = pf_torch(torch.as_tensor(matrix), method="LTL")
    pf_ltl_jax   = pf_jax(matrix)

    print("LTL J: ", pf_ltl_torch)
    print("LTL T: ", pf_ltl_jax)
    
    assert np.allclose(pf_ltl_jax, pf_ltl_torch.cpu())



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


