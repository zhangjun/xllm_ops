import ctypes
import os
import torch

if os.path.exists("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12"):
    ctypes.CDLL(
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
        mode=ctypes.RTLD_GLOBAL,
    )

from xllm_ops import common_ops
from xllm_ops.gemm import cublas_grouped_gemm
