import torch


def pack_int4_groupwise(
    W_snapped: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    OUT, IN = W_snapped.shape
    device = W_snapped.device
    PACK_IN = (IN + 7) // 8

    scales_per_elem = scales.repeat_interleave(group_size, dim=1)
    int_vals = (W_snapped / scales_per_elem).round().clamp(-7, 7).to(torch.int32)
    uint_vals = int_vals & 0xF

    if IN % 8 != 0:
        pad = PACK_IN * 8 - IN
        uint_vals = torch.cat(
            [uint_vals, torch.zeros(OUT, pad, dtype=torch.int32, device=device)],
            dim=1,
        )

    uint_vals = uint_vals.reshape(OUT, PACK_IN, 8)
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    packed = (uint_vals << shifts).sum(dim=-1).to(torch.int32)

    w_scales = (scales * 7.0).to(torch.float16)
    return packed, w_scales


def dequantize_groupwise_int4(
    packed: torch.Tensor,
    absmaxs: torch.Tensor,
    n_cols: int,
    group_size: int = 128,
) -> torch.Tensor:
    device = packed.device

    shifts = (torch.arange(8, device=device, dtype=torch.int32) * 4).view(1, 1, 8)
    vals = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF

    vals = torch.where(vals >= 8, vals - 16, vals).to(torch.float32)
    vals = vals.reshape(packed.shape[0], packed.shape[1] * 8)[:, :n_cols]

    scale_per_elem = (absmaxs.to(torch.float32) / 7.0).repeat_interleave(group_size, dim=1)
    return vals * scale_per_elem


def gptq_pack(
    W: torch.Tensor,
    H: torch.Tensor,
    group_size: int = 128,
    damp: float = 0.01,
    act_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    W = W.float().cuda()
    H = H.float().cuda()
    OUT, IN = W.shape
    assert IN % group_size == 0, f"IN ({IN}) must be divisible by group_size ({group_size})"
    n_groups = IN // group_size

    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm, :][:, perm]

    H = H + damp * H.diag().mean() * torch.eye(IN, device=H.device)
    L = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True
    )

    scales = torch.zeros((OUT, n_groups), device=W.device, dtype=W.dtype)

    for i1 in range(0, IN, group_size):
        i2 = i1 + group_size
        W1 = W[:, i1:i2].clone()
        L1 = L[i1:i2, i1:i2]
        Err = torch.zeros_like(W1)

        g_idx = i1 // group_size
        scales[:, g_idx] = (W1.abs().amax(dim=-1) / 7.0).clamp(min=1e-8)
        scale_j = scales[:, g_idx:g_idx + 1]  # (OUT, 1)

        for j in range(group_size):
            col = W1[:, j:j + 1]
            q = (col / scale_j).round().clamp(-7, 7) * scale_j
            err = (col - q) / L1[j, j]
            W1[:, j:] -= err * L1[j:j + 1, j:]
            Err[:, j:j + 1] = err

        W[:, i1:i2] = W1
        W[:, i2:] -= Err @ L[i1:i2, i2:]

    packed, w_scales = pack_int4_groupwise(W, scales, group_size)
    return packed, w_scales, perm
