import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rowwise_int4_asym_kernel(
    input_ptr,
    packed_output_ptr,
    scales_ptr,
    zero_points_ptr,
    row_width,
    BLOCK_SIZE: tl.constexpr,
    PADDED_ROW_WIDTH: tl.constexpr,
):
    """
    Triton-кернел для построчной асимметричной квантизации FP16 -> INT4.

    Проход 1: ищем row_min и row_max по строке.
    Проход 2: квантуем и упаковываем 8 значений в int32.
    """
    row_idx = tl.program_id(0)
    input_row_offset = row_idx * row_width
    num_blocks = tl.cdiv(PADDED_ROW_WIDTH, BLOCK_SIZE)

    row_min = 1e9
    row_max = -1e9

    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid_mask = col_offsets < row_width

        values = tl.load(
            input_ptr + input_row_offset + col_offsets,
            mask=valid_mask,
            other=0.0,
        ).to(tl.float32)

        values_for_min = tl.where(valid_mask, values, 1e9)
        values_for_max = tl.where(valid_mask, values, -1e9)

        row_min = tl.minimum(row_min, tl.min(values_for_min, axis=0))
        row_max = tl.maximum(row_max, tl.max(values_for_max, axis=0))

    scale = (row_max - row_min) / 15.0
    scale = tl.where(scale == 0.0, 1.0, scale)

    zero_point_f = tl.clamp(-row_min / scale, 0.0, 15.0)
    zero_point = tl.extra.cuda.libdevice.llrint(zero_point_f).to(tl.int32)

    tl.store(scales_ptr + row_idx, scale.to(scales_ptr.dtype.element_ty))
    tl.store(zero_points_ptr + row_idx, zero_point)

    scale_inv = 1.0 / scale
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

        quantized = values * scale_inv + zero_point.to(tl.float32)
        quantized = tl.clamp(quantized, 0.0, 15.0)
        quantized = tl.extra.cuda.libdevice.llrint(quantized).to(tl.int32)
        quantized = tl.where(valid_mask, quantized, 0)

        quantized_u32 = (quantized & 0xF).to(tl.uint32)
        bit_shifts = tl.arange(0, 8).to(tl.uint32) * 4
        packed_value = tl.sum(quantized_u32 << bit_shifts, axis=0).to(tl.int32)

        tl.store(packed_output_ptr + packed_row_offset + pack_idx, packed_value)


def quantize_rowwise_int4_asym(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch-обёртка над _quantize_rowwise_int4_asym_kernel.

    Возвращает:
        packed: (num_rows, ceil(row_width/8)), int32
        scales: (num_rows,), float16
        zero_points: (num_rows,), int32
    """
    assert x.is_cuda
    assert x.ndim == 2

    num_rows, row_width = x.shape
    padded_row_width = triton.next_power_of_2(row_width)
    packed_row_width = (row_width + 7) // 8

    scales = torch.empty((num_rows,), device=x.device, dtype=torch.float16)
    zero_points = torch.empty((num_rows,), device=x.device, dtype=torch.int32)
    packed_output = torch.empty(
        (num_rows, packed_row_width), device=x.device, dtype=torch.int32
    )

    BLOCK_SIZE = 32
    _quantize_rowwise_int4_asym_kernel[(num_rows,)](
        x,
        packed_output,
        scales,
        zero_points,
        row_width,
        BLOCK_SIZE,
        PADDED_ROW_WIDTH=padded_row_width,
    )

    return packed_output, scales, zero_points


def quantize_rowwise_int4_asym_ref(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch-реализация асимметричной построчной квантизации FP16 -> INT4.

    Асимметричная схема (в полном диапазоне [0, 15]):
        scale = (row_max - row_min) / 15
        zero_point = clamp(round(-row_min / scale), 0, 15)
        q = clamp(round(w / scale + zero_point), 0, 15)

    Возвращает:
        packed: (num_rows, ceil(row_width/8)), int32
        scales: (num_rows,), float16
        zero_points: (num_rows,), int32
    """
    num_rows, row_width = weight.shape

    w = weight.float()
    row_min = w.min(dim=1).values
    row_max = w.max(dim=1).values

    scale = (row_max - row_min) / 15.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    zero_point = torch.clamp(torch.round(-row_min / scale), 0, 15).to(torch.int32)

    q = torch.clamp(
        torch.round(w / scale.unsqueeze(1) + zero_point.float().unsqueeze(1)),
        0, 15,
    ).to(torch.int32)

    packed_row_width = (row_width + 7) // 8
    pad = packed_row_width * 8 - row_width
    if pad > 0:
        q = torch.nn.functional.pad(q, (0, pad), value=0)

    q = q.reshape(num_rows, packed_row_width, 8)
    shifts = torch.arange(8, device=weight.device, dtype=torch.int32) * 4
    packed = (q << shifts).sum(dim=-1).to(torch.int32)

    return packed, scale.to(torch.float16), zero_point


def dequantize_rowwise_int4_asym_ref(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """
    Reference PyTorch-деквантизация для асимметричного INT4.

    w_hat = (q - zero_point) * scale
    """
    n_rows, packed_cols = packed.shape
    device = packed.device

    shifts = (torch.arange(8, device=device, dtype=torch.int32) * 4).view(1, 1, 8)
    q = (packed.unsqueeze(-1) >> shifts) & 0xF
    q = q.reshape(n_rows, packed_cols * 8)[:, :in_features].float()

    w_hat = (q - zero_points.float().unsqueeze(1)) * scales.float().unsqueeze(1)
    return w_hat
