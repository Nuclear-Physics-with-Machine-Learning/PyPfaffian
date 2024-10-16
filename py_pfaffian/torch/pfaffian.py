import torch


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
        return pfaffian_LTL(M)
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
    