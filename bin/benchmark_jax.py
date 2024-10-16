import time
import jax


from py_pfaffian.jax import pfaffian

batch_size = 50

sizes = [2 * 2**n for n in range(8)]
# sizes = [2**n for n in range(4)]

def gen_matrix(key, size):

    
    matrix = jax.random.uniform(key, (size, size))

    # Antisymmetrization:
    matrix = matrix - matrix.T
    return matrix

gen_matrix_v = jax.vmap(gen_matrix, in_axes=(0,None))

# Single matrix workloads:
for size in sizes:

    key = jax.random.PRNGKey(int(time.time()))
    matrix = gen_matrix(key, size)

    bench_fn = jax.jit(lambda m : pfaffian(m, method="LTL"))

    pf_ltl = bench_fn(matrix)

    times = []

    for i in range(10):
        start  = time.time()
        output = bench_fn(matrix)
        output.block_until_ready()
        end    = time.time()
        times.append(end - start)

    times = jax.numpy.asarray(times[1:])
    print(size, ": ", times.mean())


# Multi matrix workloads:
for size in sizes:

    seeds = jax.numpy.asarray(
        [ jax.random.PRNGKey(int(time.time())) for _ in range(batch_size)]
    )

    matrix_v = gen_matrix_v(seeds, size)

    print(matrix_v.shape)

    bench_fn = jax.jit(jax.vmap(pfaffian, in_axes=(0,)))

    pf_ltl = bench_fn(matrix_v)

    times = []

    for i in range(10):
        start  = time.time()
        output = bench_fn(matrix_v)
        output.block_until_ready()
        end    = time.time()
        times.append(end - start)

    # print(times)
    times = jax.numpy.asarray(times[1:])
    print(size, ": ", times.mean())