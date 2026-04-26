import time

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

from triton_kernels.quantize_kernel_gptq import gptq_pack
from triton_kernels.quantized_linear import GPTQLinear, TiedQuantizedEmbedding
from utils.cuda import synchronize
from utils.memory import bytes_to_mib, gptq_linear_weight_bytes, linear_weight_bytes


def load_calibration_data(
    tokenizer,
    n_samples: int = 32,
    seq_len: int = 2048,
) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())

    old_max = tokenizer.model_max_length
    tokenizer.model_max_length = 10 ** 9
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    tokenizer.model_max_length = old_max

    total_needed = n_samples * seq_len
    if input_ids.shape[1] < total_needed:
        raise ValueError(
            f"WikiText-2 train has only {input_ids.shape[1]} tokens, "
            f"but {total_needed} are needed ({n_samples} × {seq_len})."
        )

    calib = input_ids[:, :total_needed].reshape(n_samples, seq_len).cuda()
    return calib


class _CatchDone(Exception):
    """Raised by _Catcher to stop the forward pass after catching the first input."""


class _Catcher(nn.Module):
    """Wraps the first decoder block to capture its input activations."""

    def __init__(self, module: nn.Module, inps: list, kwargs_list: list):
        super().__init__()
        self.module = module
        self.inps = inps
        self.kwargs_list = kwargs_list

    def forward(self, hidden_states, **kwargs):
        self.inps.append(hidden_states.detach())
        self.kwargs_list.append(
            {k: v for k, v in kwargs.items() if k != "past_key_value"}
        )
        raise _CatchDone


def _make_hessian_hook(name: str, Hs: dict, cnts: dict):
    """Returns a forward hook that accumulates X^T X for the named layer."""
    def hook(mod: nn.Module, inp, out):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        if name not in Hs:
            Hs[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
            cnts[name] = 0
        Hs[name] += x.T @ x
        cnts[name] += x.shape[0]
    return hook


def _replace_linear_in_block(
    block: nn.Module,
    Hs: dict,
    cnts: dict,
    group_size: int,
    act_order: bool,
    use_autotuned: bool,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    """GPTQ-quantize every nn.Linear inside ``block`` that has a Hessian."""
    for name, mod in list(block.named_modules()):
        if not (isinstance(mod, nn.Linear) and name in Hs):
            continue

        H = Hs[name] / cnts[name]
        w_packed, w_scales, perm = gptq_pack(
            mod.weight.data, H,
            group_size=group_size,
            act_order=act_order,
        )

        parts = name.split(".")
        parent = block
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(
            parent,
            parts[-1],
            GPTQLinear(
                mod, w_packed, w_scales, perm,
                group_size=group_size,
                use_autotuned=use_autotuned,
                block_m=block_m, block_n=block_n, block_k=block_k,
            ),
        )


def gptq_quantize_model(
    model: nn.Module,
    calib: torch.Tensor,
    group_size: int = 128,
    act_order: bool = True,
    use_autotuned: bool = True,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
) -> dict:
    linear_weight_bytes_before = linear_weight_bytes(model)
    linear_count_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    synchronize()
    start = time.perf_counter()

    layers = model.model.layers
    inps: list[torch.Tensor] = []
    kwargs_cache: list[dict] = []

    orig_first = layers[0]
    layers[0] = _Catcher(orig_first, inps, kwargs_cache)
    with torch.no_grad():
        for i in range(calib.shape[0]):
            try:
                model(calib[i:i + 1], use_cache=False)
            except _CatchDone:
                pass
    layers[0] = orig_first

    for k in tqdm(range(len(layers)), desc="GPTQ (group-wise)"):
        block = layers[k]
        Hs: dict[str, torch.Tensor] = {}
        cnts: dict[str, int] = {}

        hooks = [
            mod.register_forward_hook(_make_hessian_hook(name, Hs, cnts))
            for name, mod in block.named_modules()
            if isinstance(mod, nn.Linear)
        ]

        with torch.no_grad():
            for inp, kw in zip(inps, kwargs_cache):
                block(inp, **kw)

        for h in hooks:
            h.remove()

        _replace_linear_in_block(block, Hs, cnts, group_size, act_order, use_autotuned, block_m, block_n, block_k)

        new_inps: list[torch.Tensor] = []
        with torch.no_grad():
            for inp, kw in zip(inps, kwargs_cache):
                out = block(inp, **kw)
                new_inps.append(out[0] if isinstance(out, tuple) else out)
        inps = new_inps

    final_norm = model.model.norm
    H_lm: torch.Tensor | None = None
    cnt_lm = 0
    with torch.no_grad():
        for inp in inps:
            x = final_norm(inp).reshape(-1, inp.shape[-1]).float()
            if H_lm is None:
                H_lm = torch.zeros(x.shape[1], x.shape[1], device=x.device)
            H_lm = H_lm + x.T @ x
            cnt_lm += x.shape[0]

    lm = model.lm_head
    w_packed, w_scales, _ = gptq_pack(
        lm.weight.data, H_lm / cnt_lm,
        group_size=group_size,
        act_order=False,
    )
    model.lm_head = GPTQLinear(
        lm, w_packed, w_scales, perm=None,
        group_size=group_size,
        use_autotuned=use_autotuned,
        block_m=block_m, block_n=block_n, block_k=block_k,
    )

    if getattr(model.config, "tie_word_embeddings", False):
        orig_embed = model.model.embed_tokens
        model.model.embed_tokens = TiedQuantizedEmbedding(
            source=model.lm_head,
            num_embeddings=orig_embed.num_embeddings,
            embedding_dim=orig_embed.embedding_dim,
        )

    synchronize()
    quantization_time_s = time.perf_counter() - start

    linear_weight_bytes_after = gptq_linear_weight_bytes(model)

    return {
        "quantization_method": "gptq",
        "group_size": group_size,
        "act_order": act_order,
        "calib_n_samples": calib.shape[0],
        "linear_before": linear_count_before,
        "linear_after": sum(1 for m in model.modules() if isinstance(m, nn.Linear)),
        "gptq_linear_after": sum(1 for m in model.modules() if isinstance(m, GPTQLinear)),
        "quantization_time_s": quantization_time_s,
        "linear_weight_mib_before": bytes_to_mib(linear_weight_bytes_before),
        "linear_weight_mib_after": bytes_to_mib(linear_weight_bytes_after),
        "linear_weight_compression": (
            linear_weight_bytes_before / linear_weight_bytes_after
            if linear_weight_bytes_after > 0
            else 0.0
        ),
    }
