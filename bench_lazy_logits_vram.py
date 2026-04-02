import argparse
import math
import time

import torch
import torch.nn.functional as F


def bytes_to_gib(x: int) -> float:
    return x / (1024 ** 3)


def tensor_bytes(numel: int, dtype: torch.dtype) -> int:
    # PyTorch element_size() is available only for actual tensors; keep a small mapping here.
    if dtype in (torch.float16, torch.bfloat16):
        return numel * 2
    if dtype == torch.float32:
        return numel * 4
    if dtype == torch.float64:
        return numel * 8
    raise ValueError(f"Unsupported dtype for size estimate: {dtype}")


def run_once(head: torch.nn.Module, x: torch.Tensor, is_text: torch.Tensor, labels: torch.Tensor, mode: str) -> tuple[float, int]:
    assert mode in ("baseline_full_logits", "lazy_text_only")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    if mode == "baseline_full_logits":
        logits_full = head(x)  # [T, V]
        logits = logits_full[is_text]  # [T_text, V]
    else:
        logits = head(x[is_text])  # [T_text, V]

    loss = F.cross_entropy(logits, labels)
    loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
    else:
        peak = 0

    dt = time.perf_counter() - t0
    return dt, peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--vocab", type=int, default=100352)
    ap.add_argument("--text-frac", type=float, default=0.5, help="Fraction of tokens treated as text (rest treated as image tokens).")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for a meaningful VRAM benchmark.")

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    device = torch.device("cuda")
    torch.manual_seed(0)

    head = torch.nn.Linear(args.d_model, args.vocab, bias=False, device=device, dtype=dtype)

    x = torch.randn(args.seq_len, args.d_model, device=device, dtype=dtype, requires_grad=True)
    # Deterministic mask: first K tokens are text, remaining are "image" tokens.
    k = max(1, min(args.seq_len, int(round(args.seq_len * args.text_frac))))
    is_text = torch.zeros(args.seq_len, device=device, dtype=torch.bool)
    is_text[:k] = True

    labels = torch.randint(0, args.vocab, (k,), device=device, dtype=torch.long)

    # Sanity: row-wise Linear means these are exactly equal.
    with torch.no_grad():
        a = head(x)[is_text]
        b = head(x[is_text])
        if not torch.allclose(a, b, atol=0, rtol=0):
            raise SystemExit("Unexpected mismatch: Linear(head(x)[mask]) != Linear(head(x[mask])).")

    # Rough size estimate for the full logits tensor.
    full_logits_elems = args.seq_len * args.vocab
    full_logits_bytes = tensor_bytes(full_logits_elems, dtype)

    print(f"seq_len={args.seq_len} d_model={args.d_model} vocab={args.vocab} dtype={dtype} text_tokens={k}")
    print(f"Full logits tensor size (forward): {bytes_to_gib(full_logits_bytes):.3f} GiB")

    # Baseline: materialize full logits, then index to text tokens.
    head.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    dt_full, peak_full = run_once(head, x, is_text, labels, mode="baseline_full_logits")

    # Lazy: only compute logits for the text rows.
    head.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    torch.cuda.empty_cache()
    dt_lazy, peak_lazy = run_once(head, x, is_text, labels, mode="lazy_text_only")

    print(f"baseline_full_logits: time={dt_full:.4f}s peak_alloc={bytes_to_gib(peak_full):.3f} GiB")
    print(f"lazy_text_only:      time={dt_lazy:.4f}s peak_alloc={bytes_to_gib(peak_lazy):.3f} GiB")
    if peak_full > 0 and peak_lazy > 0:
        print(f"peak_saved:         {bytes_to_gib(max(0, peak_full - peak_lazy)):.3f} GiB")


if __name__ == "__main__":
    main()
