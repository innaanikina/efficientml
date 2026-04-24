import torch
import torch.nn as nn

from triton_kernels.matmul_int4 import matmul_x16_w4, matmul_x16_w4_autotuned
from triton_kernels.quantize_kernel import quantize_rowwise_int4


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        w_packed: torch.Tensor,
        w_scales: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None = None,
        use_autotuned: bool = True,
        block_m: int = 32,
        block_n: int = 32,
        block_k: int = 32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_autotuned = use_autotuned
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

        self.register_buffer("w_packed", w_packed)
        self.register_buffer("w_scales", w_scales)
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        use_autotuned: bool = True,
        block_m: int = 32,
        block_n: int = 32,
        block_k: int = 32,
    ) -> "QuantizedLinear":
        weight = linear.weight.detach().to(dtype=torch.float16)
        w_packed, w_scales = quantize_rowwise_int4(weight)
        bias = None if linear.bias is None else linear.bias.detach().to(dtype=torch.float16)

        return cls(
            w_packed=w_packed,
            w_scales=w_scales,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=bias,
            use_autotuned=use_autotuned,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)
        if self.use_autotuned:
            y = matmul_x16_w4_autotuned(
                x_2d,
                self.w_packed,
                self.w_scales,
                self.in_features,
            )
        else:
            y = matmul_x16_w4(
                x_2d,
                self.w_packed,
                self.w_scales,
                self.in_features,
                self.block_m,
                self.block_n,
                self.block_k,
            )

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*original_shape, self.out_features)


def replace_linear_layers(
    module: nn.Module,
    skip_module_names: set[str] | None = None,
    use_autotuned: bool = True,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
    _prefix: str = "",
) -> nn.Module:
    skip_module_names = skip_module_names or set()

    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name
        if name in skip_module_names or full_name in skip_module_names:
            continue

        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                QuantizedLinear.from_linear(
                    child,
                    use_autotuned=use_autotuned,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                ),
            )
        else:
            replace_linear_layers(
                child,
                skip_module_names=skip_module_names,
                use_autotuned=use_autotuned,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                _prefix=full_name,
            )

    return module
