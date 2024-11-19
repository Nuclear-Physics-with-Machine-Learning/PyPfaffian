import torch

class Pfaffian(torch.autograd.Function):

    @staticmethod
    def forward(ctx, matrix):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        pf_val = pfaffian(matrix, method="LTL")
        ctx.save_for_backward(matrix, pf_val)
        return pf_val

    @staticmethod
    def jvp(ctx, tangents):

        A, primal_out = ctx.saved_tensors

        A_dot = tangents
        primal_out = pfaffian_LTL(A)
        # The primal out is Pf(A)
        # Similar to jacobi's formula,
        # 1/pf(A) * dpf(A)/dt = 1/2 tr(A^-1 dA/dt)

        # Therefore dpf(A)/dt = 1/2 pf(A) * tr(A^-1 dA/dt)
        
        # Directly compute the inverse.  Beware of instabilities with large matrices.
        A_inv = torch.linalg.inv(A)

        product = torch.matmul(A_inv, A_dot)
        product =  primal_out * product

        tangent_out = 0.5*torch.linalg.trace(product)
        return primal_out, tangent_out
    

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        matrix, pf_val = ctx.saved_tensors
        # print(matrix)
        # print(pf_val)
        # print(grad_output.shape)
        # print(grad_output)


        # print("HERE")
        m_inv = torch.inverse(matrix)

        # print(m_inv * grad_output * pf_val)


        return m_inv  * pf_val * grad_output




def pfaffian(M : torch.Tensor, method: str ="LTL"):
    """Compute and return the Pfaffian of a matrix M

    Args:
        M (jax.Array): _description_
        method (str, optional): Available methods. Defaults to "LTL" (most efficient).  Can also use direct and recursive for small matrices.

    Raises:
        Exception: If you supply a method that isn't yet supported, this will raise an exception.
        After more methods are supported, this exception will be removed.

    Returns:
        jax.Array: A single valued number in the same dtype as the matrix M.
    """

    if method == "auto":
        N = M.shape[0]
        if N % 2 == 0 or N == 2 or N == 4: method = "direct"
        else: method = "LTL"

    if method == "LTL":
        # Use the LTL method directly:
        from . utils import pfaffian_LTL
        pf = pfaffian_LTL(M)
        pf._require_grad = True
        return pf
    elif method == "direct":
        from . utils import pfaffian_direct
        return pfaffian_direct(M)
    elif method == "recursive":
        from . utils import pfaffian_recursive
        return pfaffian_recursive(M)
    else:
        raise Exception(f"Method {method} not yet supported")
    
def log_pfaffian(M : torch.Tensor, method: str ="LTL"):
    """Compute and return the log of the Pfaffian of a matrix M

    Args:
        M (jax.Array): The matrix M to compute the pfaffian on.
        method (str, optional): _description_. Defaults to "LTL".  Recursive is not available in log format.

    Raises:
        Exception: If you supply a method that isn't yet supported, this will raise an exception.
        After more methods are supported, this exception will be removed.

    Returns:
        jax.Array: A single valued number in the same dtype as the matrix M.
    """

    if method == "auto":
        N = M.shape[0]
        if N % 2 == 0 or N == 2 or N == 4: method = "direct"
        else: method = "LTL"

    if method == "LTL":
        # Use the LTL method directly:
        # Different paths, to enable direct log in complex with a single return value:
        if torch.is_complex(M):
            raise Exception("Complex Log Pfaffian direct computation not available yet.")
            return complex_log_pfaffian_LTL(M)
        else:
            from . utils import log_pfaffian_LTL
            return log_pfaffian_LTL(M)
    elif method == "direct":
        from . utils import log_pfaffian_direct
        return log_pfaffian_direct(M)
    else:
        raise Exception(f"Method {method} not yet supported")
    