import torch

from triton_kernels.quantize_kernel_sym import quantize_rowwise_int4
from triton_kernels.quantize_kernel_asym import quantize_rowwise_int4_asym
from triton_kernels.quantize_kernel_gptq import pack_int4_groupwise


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


def make_case_asym(B: int, IN: int, OUT: int, seed: int = 0, device: str = "cuda"):
    torch.manual_seed(seed)
    x = torch.randn(B, IN, device=device, dtype=torch.float16)
    w = torch.randn(OUT, IN, device=device, dtype=torch.float16)
    w_packed, w_scales, w_zp = quantize_rowwise_int4_asym(w)
    return x, w, w_packed, w_scales, w_zp


# IN кратны 128 для group-wise кернела
SHAPES_GPTQ = [
    (1, 128, 64),
    (4, 256, 128),
    (8, 128, 512),
    (16, 512, 256),
    (32, 512, 512),
    (128, 2048, 2048),
]


def make_case_gptq(B: int, IN: int, OUT: int, group_size: int = 128, seed: int = 0, device: str = "cuda"):
    torch.manual_seed(seed)
    x = torch.randn(B, IN, device=device, dtype=torch.float16)
    w = torch.randn(OUT, IN, device=device, dtype=torch.float32).cuda()
    n_groups = IN // group_size
    scales = (w.reshape(OUT, n_groups, group_size).abs().amax(dim=-1) / 7.0).clamp(min=1e-8)
    w_packed, w_scales = pack_int4_groupwise(w, scales, group_size)
    return x, w.to(torch.float16), w_packed, w_scales


def iter_cases(device: str = "cuda"):
    for i, (B, IN, OUT) in enumerate(SHAPES):
        yield (B, IN, OUT), make_case(B, IN, OUT, seed=i, device=device)
