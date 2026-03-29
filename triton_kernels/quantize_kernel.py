import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rowwise_int4_kernel(
    input_ptr,
    packed_output_ptr,
    row_absmax_ptr,
    row_width,
    BLOCK_SIZE: tl.constexpr,
    PADDED_ROW_WIDTH: tl.constexpr,
):
    """
    Triton кернель для построчной квантизации weights fp16 -> int4
    """
    row_idx = tl.program_id(0)
    input_row_offset = row_idx * row_width

    num_blocks = tl.cdiv(PADDED_ROW_WIDTH, BLOCK_SIZE)

    row_absmax = 0.0
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid_mask = col_offsets < row_width

        values = tl.load(
            input_ptr + input_row_offset + col_offsets,
            mask=valid_mask,
            other=0.0,
        ).to(tl.float32)

        block_absmax = tl.max(tl.abs(values), axis=0)
        row_absmax = tl.maximum(row_absmax, block_absmax)

    tl.store(row_absmax_ptr + row_idx, row_absmax.to(row_absmax_ptr.dtype.element_ty))

    scale_inv = tl.where(row_absmax > 0, 1.0 / row_absmax, 0.0)

    packed_row_width = tl.cdiv(row_width, 8)
    packed_row_offset = row_idx * packed_row_width

    for pack_idx in range(packed_row_width):
        col_offsets = pack_idx * 8 + tl.arange(0, 8)
        valid_mask = col_offsets < row_width

        values = tl.load(
            input_ptr + input_row_offset + col_offsets,
            mask=valid_mask,
            other=0.0,
        ).to(tl.float32)

        quantized = values * scale_inv * 7
        quantized = tl.clamp(quantized, -7, 7)
        quantized = tl.extra.cuda.libdevice.llrint(quantized).to(tl.int32)
        quantized = tl.where(valid_mask, quantized, 0)

        quantized_u32 = (quantized & 0xF).to(tl.uint32)
        bit_shifts = tl.arange(0, 8).to(tl.uint32) * 4

        packed_value = tl.sum(quantized_u32 << bit_shifts, axis=0).to(tl.int32)

        tl.store(packed_output_ptr + packed_row_offset + pack_idx, packed_value)


def quantize_rowwise_int4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pytorch функция (обертка над _quantize_rowwise_int4_kernel)
    """
    assert x.is_cuda
    assert x.ndim == 2

    num_rows, row_width = x.shape
    padded_row_width = triton.next_power_of_2(row_width)
    packed_row_width = (row_width + 7) // 8

    row_absmax = torch.empty((num_rows,), device=x.device, dtype=torch.float16)
    packed_output = torch.empty(
        (num_rows, packed_row_width),
        device=x.device,
        dtype=torch.int32,
    )
    
    BLOCK_SIZE = 32
    _quantize_rowwise_int4_kernel[(num_rows,)](
        x,
        packed_output,
        row_absmax,
        row_width,
        BLOCK_SIZE,
        PADDED_ROW_WIDTH=padded_row_width,
    )

    return packed_output, row_absmax


def dequantize_rowwise_int4(
    packed: torch.Tensor, 
    absmaxs: torch.Tensor, 
    n_cols: int
) -> torch.Tensor:
    """
    Обычная pytorch-функция для деквантизации для тестирования
    """

    n_rows, packed_cols = packed.shape
    device = packed.device

    shifts = (torch.arange(8, device=device, dtype=torch.int32) * 4).view(1, 1, 8)
    vals = (packed.unsqueeze(-1) >> shifts) & 0xF

    vals = vals.to(torch.int32)
    vals = torch.where(vals >= 8, vals - 16, vals).to(torch.float32)

    vals = vals.reshape(n_rows, packed_cols * 8)[:, :n_cols]
    scale = (absmaxs.to(torch.float32) / 7.0).unsqueeze(1)
    x_hat = vals * scale
    return x_hat


def compute_quantization_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    diff = original.float() - reconstructed.float()
    
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_abs_error = diff.abs().max().item()

    return {
        "mse": mse,
        "mae": mae,
        "max_abs_error": max_abs_error,
    }


def get_memory_bytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.numel()


def run_benchmark():
    
    torch.manual_seed(42)
    
    test_matrices = {
        "64x64": torch.randn(64, 64, device="cuda", dtype=torch.float16),
        "128x128": torch.randn(128, 128, device="cuda", dtype=torch.float16),
        "256x256": torch.randn(256, 256, device="cuda", dtype=torch.float16),
        "512x512": torch.randn(256, 256, device="cuda", dtype=torch.float16),
        "1024x1024": torch.randn(1024, 1024, device="cuda", dtype=torch.float16),
        "4096x4096": torch.randn(4096, 4096, device="cuda", dtype=torch.float16),
    }
    
    print("=" * 80)
    print("INT4 Row-wise Quantization Benchmark")
    print("=" * 80)
    
    for name, x in test_matrices.items():
        print(f"\n{name}")
        print("-" * 60)
        
        packed, absmaxs = quantize_rowwise_int4(x)
        x_hat = dequantize_rowwise_int4(packed, absmaxs, x.shape[1])
        
        original_bytes = get_memory_bytes(x)
        packed_bytes = get_memory_bytes(packed) + get_memory_bytes(absmaxs)
        compression_ratio = original_bytes / packed_bytes
        
        print(f"Shape: {x.shape[0]} x {x.shape[1]}")
        print(f"Original:   {original_bytes:>12,} bytes ({original_bytes / 1024 / 1024:.2f} MB)")
        print(f"Quantized:  {packed_bytes:>12,} bytes ({packed_bytes / 1024 / 1024:.2f} MB)")
        print(f"Compression: {compression_ratio:.2f}x")
        
        errors = compute_quantization_error(x, x_hat)
        print(f"\nQuantization Error:")
        print(f"  MSE:           {errors['mse']:.6f}")
        print(f"  MAE:           {errors['mae']:.6f}")
        print(f"  Max Abs Error: {errors['max_abs_error']:.6f}")
    
    print("\n" + "=" * 80)
    print("Benchmark completed")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
