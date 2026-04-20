import torch

from triton_kernels.quantize_kernel import quantize_rowwise_int4


SHAPES = [
    (1, 16, 16),
    (4, 32, 8),
    (8, 64, 64),
    (16, 128, 256),
    (32, 512, 512),
    (128, 2048, 2048),
    (4, 40, 24),
    (1, 8, 2048),
]


def make_case(B: int, IN: int, OUT: int, seed: int = 0, device: str = "cuda"):
    torch.manual_seed(seed)
    x = torch.randn(B, IN, device=device, dtype=torch.float16)
    w = torch.randn(OUT, IN, device=device, dtype=torch.float16)
    w_packed, w_scales = quantize_rowwise_int4(w)
    return x, w, w_packed, w_scales


def iter_cases(device: str = "cuda"):
    for i, (B, IN, OUT) in enumerate(SHAPES):
        yield (B, IN, OUT), make_case(B, IN, OUT, seed=i, device=device)
