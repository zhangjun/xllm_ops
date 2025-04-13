from typing import Dict, Tuple

import torch

def get_cuda_stream() -> int:
    return torch.cuda.current_stream().cuda_stream
