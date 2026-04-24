import torch
import torch.nn as nn

from triton_kernels.quantize_kernel import quantize_rowwise_int4, dequantize_rowwise_int4
from triton_kernels.matmul_kernel import matmul_x16_w4, matmul_x16_w4_ref


class QuantizedLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, w_packed: torch.Tensor, 
        w_scales: torch.Tensor, bias: torch.Tensor | None = None, use_triton: bool = True
        ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_triton = use_triton

        self.register_buffer("w_packed", w_packed)
        self.register_buffer("w_scales", w_scales)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    

