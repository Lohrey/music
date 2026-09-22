"""Independent check: does MPS half-precision SDPA decode (q_len=1, kv>=~1024) corrupt?

Mirrors yue2.modeling_yue2.sdpa's MPS path (GQA expanded via repeat_interleave,
F.scaled_dot_product_attention with no mask) but with deterministic inputs and a
float64 CPU reference.  No model weights are loaded; tensors are tiny.

A row counts as BAD when it is wrong, non-finite, **or not reproducible**: the
same inputs must return the same bytes twice.  Nondeterminism is itself the
signal on the defective kernel (kv=1024 can look correct once and then differ
wildly), so the repeat call is part of the verdict, not just diagnostics.

Usage: python tools/mps_sdpa_check.py           # exit 1 if the defect is present
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

DEV = torch.device("mps")
H, KVH, D = 16, 8, 128  # yue2-like: 16 query heads, 8 kv heads, head_dim 128
BIG = 9500  # static-cache capacity, like the model
TOL = 0.05


def tensors(kv: int, dtype, layout: str, seed: int = 1234):
    """(q, k, v) on MPS, deterministic; `layout` mirrors how the cache is built."""
    gen = torch.Generator().manual_seed(seed)
    q_cpu = torch.randn(1, H, 1, D, generator=gen)
    k_cpu = torch.randn(1, KVH, kv, D, generator=gen)
    v_cpu = torch.randn(1, KVH, kv, D, generator=gen)
    if layout == "exact":
        k = k_cpu.to(device=DEV, dtype=dtype)
        v = v_cpu.to(device=DEV, dtype=dtype)
    else:  # large zero buffer + sliced view (the model's cache)
        k = torch.zeros(1, KVH, BIG, D, dtype=dtype, device=DEV)
        v = torch.zeros(1, KVH, BIG, D, dtype=dtype, device=DEV)
        k[:, :, :kv] = k_cpu.to(device=DEV, dtype=dtype)
        v[:, :, :kv] = v_cpu.to(device=DEV, dtype=dtype)
        k = k[:, :, :kv]
        v = v[:, :, :kv]
    # GQA expansion happens on the device in the model's MPS fallback
    k = k.repeat_interleave(H // KVH, dim=1)
    v = v.repeat_interleave(H // KVH, dim=1)
    return q_cpu.to(device=DEV, dtype=dtype), k, v


def attention(q, k, v, variant: str):
    if variant == "plain":
        return F.scaled_dot_product_attention(q, k, v)
    if variant == "zero_mask":
        mask = torch.zeros(1, 1, 1, k.shape[-2], dtype=q.dtype, device=q.device)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if variant == "fp32_attn":
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(q.dtype)
    if variant == "math":
        return ((q.float() @ k.float().transpose(-1, -2)) * D**-0.5).softmax(-1).to(q.dtype) @ v
    raise ValueError(variant)


def measure(kv: int, dtype, layout: str = "view", variant: str = "plain"):
    q, k, v = tensors(kv, dtype, layout)
    out = attention(q, k, v, variant)
    torch.mps.synchronize()
    out2 = attention(q, k, v, variant)  # same inputs again: deterministic?
    torch.mps.synchronize()
    ref = F.scaled_dot_product_attention(
        q.float().cpu().double(), k.float().cpu().double(), v.float().cpu().double()
    )
    got = out.float().cpu()
    err = float((got - ref).abs().max())
    nonfinite = int((~torch.isfinite(got)).sum())
    repeat_err = float((out2.float().cpu() - got).abs().max())
    return err, nonfinite, repeat_err


def verdict(err: float, nf: int, rep: float) -> str:
    """BAD when wrong, non-finite, or not reproducible (see module docstring)."""
    return "OK " if (err <= TOL and nf == 0 and rep <= TOL) else "BAD"


def main() -> int:
    print(f"torch {torch.__version__} | mps available: {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        print("no MPS; nothing to check")
        return 0
    defect = False

    print("\n-- kv sweep, bf16, model-like sliced view --")
    for kv in (512, 1000, 1023, 1024, 1025, 1100, 2048):
        err, nf, rep = measure(kv, torch.bfloat16, "view")
        flag = verdict(err, nf, rep)
        if flag == "BAD":
            defect = True
        print(
            f"   kv={kv:5d}  {flag}  max|err|={err:9.4f}  non-finite={nf:3d}  rerun-diff={rep:8.4f}"
        )

    print("\n-- dtype comparison at kv=1024 (sliced view) --")
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        err, nf, rep = measure(1024, dtype, "view")
        flag = verdict(err, nf, rep)
        if flag == "BAD" and dtype != torch.float32:
            defect = True
        print(
            f"   {str(dtype)[6:]:8s} {flag}  max|err|={err:9.4f}  non-finite={nf:3d}  rerun-diff={rep:8.4f}"
        )

    print("\n-- layout comparison, bf16, kv=1024 --")
    for layout in ("view", "exact"):
        err, nf, rep = measure(1024, torch.bfloat16, layout)
        flag = verdict(err, nf, rep)
        print(
            f"   {layout:6s} {flag}  max|err|={err:9.4f}  non-finite={nf:3d}  rerun-diff={rep:8.4f}"
        )

    print("\n-- workarounds at kv=2048, bf16, sliced view --")
    for variant in ("plain", "zero_mask", "fp32_attn", "math"):
        err, nf, rep = measure(2048, torch.bfloat16, "view", variant)
        flag = verdict(err, nf, rep)
        print(
            f"   {variant:10s} {flag}  max|err|={err:9.4f}  non-finite={nf:3d}  rerun-diff={rep:8.4f}"
        )

    print("\n-- enable_gqa=True (no repeat_interleave), bf16, kv=2048 --")
    try:
        gen = torch.Generator().manual_seed(7)
        q = torch.randn(1, H, 1, D, generator=gen).to(DEV, torch.bfloat16)
        k = torch.randn(1, KVH, 2048, D, generator=gen).to(DEV, torch.bfloat16)
        v = torch.randn(1, KVH, 2048, D, generator=gen).to(DEV, torch.bfloat16)
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        torch.mps.synchronize()
        ref = F.scaled_dot_product_attention(
            q.float().cpu().double(), k.float().cpu().double(), v.float().cpu().double()
        )
        err = float((out.float().cpu() - ref).abs().max())
        print(f"   enable_gqa {'OK ' if err <= TOL else 'BAD'}  max|err|={err:9.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"   enable_gqa not available on this build: {exc}")

    print("\nDEFECT PRESENT" if defect else "\ndefect not reproduced")
    return 1 if defect else 0


if __name__ == "__main__":
    sys.exit(main())
