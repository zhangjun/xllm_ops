from typing import List, Optional, Tuple

import torch
from xllm_ops.utils import get_cuda_stream

def cublas_grouped_gemm(
    inputs: List[torch.Tensor],
    weights: List[torch.Tensor],
    outputs: List[torch.Tensor],
    out_dtype: torch.dtype,
) -> None:
    assert (
        len(inputs) > 0 and len(weights) > 0 and len(outputs) > 0
    ), "Inputs/weights/outputs should not be empty!"
    cublas_handle = torch.cuda.current_blas_handle()
    torch.ops.xllm_ops.cublas_grouped_gemm.default(
        inputs,
        weights,
        outputs,
        out_dtype,
        cublas_handle,
        get_cuda_stream(),
    )