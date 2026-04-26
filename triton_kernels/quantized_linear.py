import torch
import torch.nn as nn

from triton_kernels.matmul_int4_gptq import matmul_x16_w4_groupwise
from triton_kernels.quantization_kernels import QuantizedWeights, get_quantized_kernel
from triton_kernels.quantize_kernel_gptq import dequantize_groupwise_int4


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        quantized_weights: QuantizedWeights,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None = None,
        use_autotuned: bool = True,
        block_m: int = 32,
        block_n: int = 32,
        block_k: int = 32,
        kernel_name: str = "rowwise_int4",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_autotuned = use_autotuned
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.kernel_name = kernel_name
        self.quantized_metadata = dict(quantized_weights.metadata)
        self.quantized_tensor_names = tuple(quantized_weights.tensors.keys())

        for name, tensor in quantized_weights.tensors.items():
            self.register_buffer(f"quantized_{name}", tensor)
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    @property
    def quantized_weights(self) -> QuantizedWeights:
        return QuantizedWeights(
            tensors={
                name: getattr(self, f"quantized_{name}")
                for name in self.quantized_tensor_names
            },
            metadata=dict(self.quantized_metadata),
        )

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        use_autotuned: bool = True,
        block_m: int = 32,
        block_n: int = 32,
        block_k: int = 32,
        kernel_name: str = "rowwise_int4",
    ) -> "QuantizedLinear":
        weight = linear.weight.detach().to(dtype=torch.float16)
        kernel = get_quantized_kernel(kernel_name)
        quantized_weights = kernel.quantize(weight)
        bias = None if linear.bias is None else linear.bias.detach().to(dtype=torch.float16)

        return cls(
            quantized_weights=quantized_weights,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=bias,
            use_autotuned=use_autotuned,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            kernel_name=kernel_name,
        )

    def dequantize_rows(self, indices: torch.Tensor) -> torch.Tensor:
        kernel = get_quantized_kernel(self.kernel_name)
        if kernel.dequantize is None:
            raise NotImplementedError(f"Kernel '{self.kernel_name}' does not support dequantize")
        selected = QuantizedWeights(
            tensors={
                name: getattr(self, f"quantized_{name}").index_select(0, indices)
                for name in self.quantized_tensor_names
            },
            metadata=dict(self.quantized_metadata),
        )
        return kernel.dequantize(selected, self.in_features).to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)
        kernel = get_quantized_kernel(self.kernel_name)
        quantized_weights = self.quantized_weights
        if self.use_autotuned and kernel.matmul_autotuned is not None:
            y = kernel.matmul_autotuned(
                x_2d,
                quantized_weights,
                self.in_features,
            )
        else:
            y = kernel.matmul(
                x_2d,
                quantized_weights,
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
    kernel_name: str = "rowwise_int4",
    _prefix: str = "",
    _root: nn.Module | None = None,
) -> nn.Module:
    skip_module_names = skip_module_names or set()
    is_top_level = _root is None
    if is_top_level:
        _root = module

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
                    kernel_name=kernel_name,
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
                kernel_name=kernel_name,
                _prefix=full_name,
                _root=_root,
            )

    if is_top_level:
        _maybe_tie_embeddings(_root)

    return module


def _maybe_tie_embeddings(model: nn.Module) -> None:
    if not getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        return

    lm_head = getattr(model, "lm_head", None)
    inner = getattr(model, "model", None)
    embed_tokens = getattr(inner, "embed_tokens", None) if inner is not None else None

    if isinstance(lm_head, QuantizedLinear) and isinstance(embed_tokens, nn.Embedding):
        inner.embed_tokens = TiedQuantizedEmbedding(
            source=lm_head,
            num_embeddings=embed_tokens.num_embeddings,
            embedding_dim=embed_tokens.embedding_dim,
        )


class GPTQLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        w_packed: torch.Tensor,
        w_scales: torch.Tensor,
        perm: torch.Tensor | None = None,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size

        self.register_buffer("w_packed", w_packed)
        self.register_buffer("w_scales", w_scales)
        self.register_buffer("perm", perm)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone().to(torch.float16))
        else:
            self.register_buffer("bias", None)

    def dequantize_rows(self, indices: torch.Tensor) -> torch.Tensor:
        packed = self.w_packed.index_select(0, indices)
        scales = self.w_scales.index_select(0, indices)
        return dequantize_groupwise_int4(
            packed, scales, self.in_features, self.group_size
        ).to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.in_features).contiguous()
        if self.perm is not None:
            x = x.index_select(1, self.perm)
        y = matmul_x16_w4_groupwise(
            x, self.w_packed, self.w_scales, self.in_features, self.group_size
        )
        y = y.reshape(*shape[:-1], self.out_features)
        return y + self.bias if self.bias is not None else y


class TiedQuantizedEmbedding(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.source = source
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1)
        out = self.source.dequantize_rows(flat)
        return out.reshape(*indices.shape, self.embedding_dim)
