from jax import jit, lax
import jax.numpy as numpy

from jax import Array

from jax import custom_jvp


def pfaffian_direct(A: Array):
    """Compute the Pfaffian directly.

    Args:
        A (jax.Array): A skew-symmetric matrix

    Raises:
        Exception: If the size of the matrix is larger than 4x4, there is an error

    Returns:
        Array: Scalar of pf(A)
    """


    n = A.shape[0]

    if n % 2 == 1: return 0.0

    if n == 2: return -A[1,0]

    if n == 4: 
        val = A[1,0]*A[3,2] - A[2,0]*A[3,1] + A[2,1]*A[3,0]
        return val
    else:
        raise Exception("Not implemented for N > 4")


def log_pfaffian_direct(A: Array):
    """Compute the log of the Pfaffian directly.

    Args:
        A (jax.Array): A skew-symmetric matrix

    Raises:
        Exception: If the size of the matrix is larger than 4x4, there is an error

    Returns:
        tuple(Array, Array): Scalar tuple of sign, log(|pf(A)|)
    """

    n = A.shape[0]
    

    if n % 2 == 1: return -1.0 , numpy.inf

    if n == 2: return numpy.sign(-A[1,0]), numpy.log(numpy.abs(A[1,0]))

    if n == 4: 
        val = A[1,0]*A[3,2] - A[2,0]*A[3,1] + A[2,1]*A[3,0]
        return numpy.sign(val), numpy.log(numpy.abs(val))
    else:
        raise Exception("Not implemented")
    
def pfaffian_recursive(A : Array):
    """Recursive Implementation of the pfaffian.  Not recommend over small matrices, but useful for correctness checking.

    Args:
        A (jax.Array): A skew-symmetric matrix

    Returns:
        Array: The value of the pfaffian
    """
    n = A.shape[0]

    if n == 2: 
        # print("Returning base case ", A[1,0])
        return -A[1,0]

    # Recurse:
    delete_0 = numpy.delete(numpy.delete(A, 0, axis=0), 0, axis=1)
    this_pf = 0.0
    for i in range(1, n):
        j =  i 

        a_0j = (-1)**(j) * A[0,j]

        # Subtracting by 1 because we already deleted a row and column:
        delete_index = j - 1

        submatrix = numpy.delete(numpy.delete(delete_0, delete_index, axis=0), delete_index, axis=1)

        this_pf += a_0j * pfaffian_recursive(submatrix)

    return -this_pf


def pivot(_A : Array, _k : int, _kp : int):
    """Perform a Pivot on the Matrix A

    Args:
        _A (Array): Matrix, A, of size nxn where n > _k, n > _kp
        _k (int): The first index to swap
        _kp (int): The second index to swap

    Returns:
        tuple(Array, float): The Matrix with the two specified rows swapped, and the sign of the operation.
    """
    
    temp = _A[_k + 1]
    _A = _A.at[_k + 1].set(_A[_kp])
    _A = _A.at[_kp].set(temp)

    # Then interchange columns _k+1 and _kp
    temp = _A[:, _k + 1]
    _A = _A.at[:, _k + 1].set(_A[:, _kp])
    _A = _A.at[:, _kp].set(temp)

    return _A, -1.0

def no_pivot(_A : Array, _k : int, _kp : int):
    """A Null operation.  Mimic the signature and return of the `pivot` function.  Always returns _A, 1.0

    Args:
        _A (Array): Matrix, A, of size nxn where n > _k, n > _kp
        _k (int): Ignored
        _kp (int): Ignored

    Returns:
        tuple(Array, float): The Matrix _A, and 1.0
    """
        
    return _A, 1.0


def form_gauss_vector(_A : Array, _k : int):
    """
    Form the gauss vector and update the pfaffian value as the return


    Args:
        _A (Array): Matrix A, from which to create the Gauss Vector
        _k (int): The index _k of the matrix, where _k < A.shape[0]

    Returns:
        tuple(Array, Array): The updated matrix _A and the update to the pfaffian value
    """

    
    pfaffian_update = _A[_k,_k+1]

    mu = _A[_k,:] / pfaffian_update
    nu = _A[:, _k+1]

    _A = _A + numpy.outer(mu, nu) - numpy.outer(nu, mu)

    return _A, pfaffian_update


# pivot = jit(pivot, donate_argnums=0)
# no_pivot = jit(no_pivot, donate_argnums=0)
# form_gauss_vector = jit(form_gauss_vector, donate_argnums=0)


@custom_jvp
def pfaffian_LTL(A : Array):
    """pfaffian_LTL(A, overwrite_a=False)

    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    the Parlett-Reid algorithm.
    """

    # JAX does not support runtime checking.
    # All checks are removed!
    n = A.shape[0]
    if n % 2 == 1: return 0.0

    pfaffian_val = 1.0

    def pfaffian_iteration(carry, _k):
        # This needs to become a function we can iterate over:

        # First, find the largest entry in A[k+1:,k] and
        # permute it to A[k+1,k]

        # Doing this with a full region and we can use a staticlly-sized
        # mask and a where operation to dynamically mask based on index:

        _A = carry[0]
        _current_pfaffian_val = carry[1]

        full_region = _A[:,_k]
        mask = numpy.arange(full_region.shape[0]) < _k
        full_region = numpy.where(mask, 0.0, full_region)

        _kp = _k + 1 + numpy.abs(full_region).argmax() - (_k+1)

        # Apply a pivot if needed:
        _A, sign = lax.cond(_kp != _k + 1, pivot, no_pivot, _A, _k, _kp)
        
        # get the update and form the gauss vectors if needed:
        _A, update = form_gauss_vector(_A, _k)

        return (_A, sign * update * _current_pfaffian_val), None

    # Possible performance optimization to reuse memory here:
    # pfaffian_iteration = jit(pfaffian_iteration, donate_argnums=0)


    # Define the matrix indexes we'll look at:
    k_list = numpy.arange(0, n-1, 2)

    carry = (
        A,
        pfaffian_val
    )

    # Perform the decomposition iteratively:
    carry, _ = lax.scan(pfaffian_iteration, carry, k_list)

    return carry[1]

@pfaffian_LTL.defjvp
def pfaffian_LTL_jvp(primals, tangents):
    A, = primals
    A_dot, = tangents
    primal_out = pfaffian_LTL(A)
    # The primal out is Pf(A)
    # Similar to jacobi's formula,
    # 1/pf(A) * dpf(A)/dt = 1/2 tr(A^-1 dA/dt)

    # Therefore dpf(A)/dt = 1/2 pf(A) * tr(A^-1 dA/dt)
    
    # Directly compute the inverse.  Beware of instabilities with large matrices.
    A_inv = numpy.linalg.inv(A)

    product = numpy.matmul(A_inv, A_dot)
    product =  primal_out * product

    tangent_out = 0.5*numpy.linalg.trace(product)
    return primal_out, tangent_out



# @custom_jvp
def log_pfaffian_LTL(A):
    """pfaffian_LTL(A, overwrite_a=False)

    Compute the log of the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). This function uses
    the Parlett-Reid algorithm.
    """

    # JAX does not support runtime checking.
    # All checks are removed!
    n = A.shape[0]

    pfaffian_sign = 1.0
    log_pfaffian  = 0.0

    if n % 2 == 1: return 1.0, -numpy.inf

    def pfaffian_iteration(carry, _k):
        # This needs to become a function we can iterate over:

        # First, find the largest entry in A[k+1:,k] and
        # permute it to A[k+1,k]

        # Doing this with a full region and we can use a staticlly-sized
        # mask and a where operation to dynamically mask based on index:

        _A = carry[0]
        _current_pfaffian_sign = carry[1]
        _current_log_pfaffian  = carry[2]

        full_region = _A[:,_k]
        _mask = numpy.arange(full_region.shape[0]) < _k

        full_region = numpy.where(_mask, 0.0, full_region)
        _kp = numpy.abs(full_region).argmax()
        # Apply a pivot if needed:
        _A, sign = lax.cond(_kp != _k + 1, pivot, no_pivot, _A, _k, _kp)

        # get the update and form the gauss vectors if needed:
        _A, update = form_gauss_vector(_A, _k)

        return (_A, sign*_current_pfaffian_sign*numpy.sign(update),  numpy.log(numpy.abs(update)) + _current_log_pfaffian), None

    pfaffian_iteration = jit(pfaffian_iteration, donate_argnums=0)


    # Define the matrix indexes we'll look at:
    k_list = numpy.arange(0, n-1, 2)

    # We compute here the log of every element in the matrix to improve numerical performance:
    carry = (
        A,
        pfaffian_sign,
        log_pfaffian
    )

    carry, _ = lax.scan(pfaffian_iteration, carry, k_list)

    return carry[1], carry[2]



# @log_pfaffian_LTL.defjvp
# def f_jvp(primals, tangents):
#     A, = primals
#     A_dot, = tangents
#     primal_out = log_pfaffian_LTL(A)
#     print(primal_out)
#     # The primal out is sign, log(Pf(A))
#     # Similar to jacobi's formula,
#     # 1/pf(A) * dpf(A)/dt = 1/2 tr(A^-1 dA/dt)

#     # Using d/dt [log(pf)] = 1/pf dpf/df, we compute the direivative of the log directly.

    

#     A_inv = numpy.linalg.inv(A)
#     print(A_inv)

#     print(numpy.matmul(A_inv, A_dot))

#     print(numpy.sum(A_inv))
#     print(numpy.linalg.trace(A_inv))
#     print("Product: ", A_inv*A_dot)

#     tangent_out = 0.5 * numpy.linalg.trace(numpy.matmul(A_inv, A_dot))
#     return primal_out, (primal_out[0], tangent_out)