import torch

from triton_kernels.quantize_kernel import (
    quantize_rowwise_int4,
    dequantize_rowwise_int4,
)


def test_output_shapes():
    x = torch.randn(4, 17, device="cuda", dtype=torch.float16)
    packed, absmaxs = quantize_rowwise_int4(x)
    
    assert packed.shape == (4, 3)
    assert packed.dtype == torch.int32
    assert absmaxs.shape == (4,)
    assert absmaxs.dtype == torch.float16

def test_roundtrip_known_values():
    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
    ], device="cuda", dtype=torch.float16)
    
    packed, absmaxs = quantize_rowwise_int4(x)
    x_hat = dequantize_rowwise_int4(packed, absmaxs, x.shape[1])
    
    assert absmaxs[0].item() == 4.0
    assert x_hat.shape == x.shape

def test_error_bounded():
    torch.manual_seed(42)
    x = torch.randn(8, 32, device="cuda", dtype=torch.float16)
    
    packed, absmaxs = quantize_rowwise_int4(x)
    x_hat = dequantize_rowwise_int4(packed, absmaxs, x.shape[1])
    
    err = (x.float() - x_hat).abs()
    max_possible_err = (absmaxs.float() / 7.0).unsqueeze(1)
    
    assert torch.all(err <= max_possible_err + 1e-3)

def test_zero_row():
    x = torch.zeros(1, 16, device="cuda", dtype=torch.float16)
    
    packed, absmaxs = quantize_rowwise_int4(x)
    x_hat = dequantize_rowwise_int4(packed, absmaxs, x.shape[1])
    
    assert absmaxs[0].item() == 0.0
    assert torch.allclose(x_hat, torch.zeros_like(x_hat))

def test_single_value_row():
    x = torch.tensor([[5.0] + [0.0] * 7], device="cuda", dtype=torch.float16)
    
    packed, absmaxs = quantize_rowwise_int4(x)
    x_hat = dequantize_rowwise_int4(packed, absmaxs, x.shape[1])
    
    assert absmaxs[0].item() == 5.0
    assert x_hat[0, 0].item() == 5.0
