import torch
import torch.nn as nn

from triton_kernels.quantized_linear import QuantizedLinear


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.numel()


def bytes_to_mib(nbytes: int) -> float:
    return nbytes / 1024**2


def tensor_memory_mib(*tensors: torch.Tensor) -> float:
    return bytes_to_mib(sum(tensor_nbytes(tensor) for tensor in tensors))


def linear_weight_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += tensor_nbytes(module.weight)
            if module.bias is not None:
                total += tensor_nbytes(module.bias)
    return total


def quantized_linear_weight_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            total += tensor_nbytes(module.w_packed)
            total += tensor_nbytes(module.w_scales)
            if module.bias is not None:
                total += tensor_nbytes(module.bias)
    return total


def linear_and_quantized_weight_bytes(model: nn.Module) -> int:
    return linear_weight_bytes(model) + quantized_linear_weight_bytes(model)
