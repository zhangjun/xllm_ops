import argparse
import torch
import triton
import triton.language as tl

# https://github.com/triton-lang/triton/tree/main/python/tutorials
# https://zhuanlan.zhihu.com/p/684473453
# https://zhuanlan.zhihu.com/p/895509143
# https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c
# https://isamu-website.medium.com/understanding-triton-tutorials-part-2-f6839ce50ae7
# http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8/OpenAI%20Triton%20MLIR%20%E7%AC%AC%E4%B8%80%E7%AB%A0%20Triton%20DSL/#0x1-triton-kernel
# https://blog.csdn.net/Kongxiangyunltj/article/details/139723840

@triton.autotune(configs=[
    triton.Config(kwargs={'TILE': 128}),
    triton.Config(kwargs={'TILE': 256})],
    key=['N']
)
@triton.jit
def _add(z_ptr, x_ptr, y_ptr, N,
    # # Meta-parameters
    TILE: tl.constexpr
):
    # same as torch.arange
    offsets = tl.arange(0, TILE)
    offsets += tl.program_id(0) * TILE
    # create TILE pointers to X, Y, Z
    x_ptrs = x_ptr + offsets
    y_ptrs = y_ptr + offsets
    z_ptrs = z_ptr + offsets
    # load TILE elements of X, Y, Z
    mask = offsets < N
    x = tl.load(x_ptrs, mask=mask)
    y = tl.load(y_ptrs, mask=mask)
    # do computations
    z = x + y
    # write-back TILE elements of X, Y, Z
    tl.store(z_ptrs, z, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    N = output.numel()
    grid = lambda args: (triton.cdiv(N, args["TILE"]), )
    _add[grid](output, x, y, N)
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=bool, default=False)
    args = parser.parse_args()
    N = 1024
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    res = add(x, y)

    if args.bench:
        benchmark.run(print_data=True, show_plots=True)