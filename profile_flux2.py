"""
Inference-only low-level profiler for Flux2 transformer checkpoints.

This script reports only values measured from actual execution:
- end-to-end forward latency
- per-block latency and memory deltas
- operator-level timing/memory via torch.profiler
- kernel-level microbenchmarks over shape sweeps

No analytic or hardcoded block-cost formulas are used.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from transformer_flux2 import Flux2Transformer2DModel


# Candidate weight filenames commonly used by Diffusers/HF exports.
WEIGHT_CANDIDATES = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "model.safetensors",
    "model.bin",
)


def parse_csv_ints(csv_str: str) -> list[int]:
    """
    Parse comma-separated integers, e.g. "256,512,1024" -> [256, 512, 1024].
    """
    if not csv_str:
        return []
    return [int(x.strip()) for x in csv_str.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    """
    Define and parse CLI arguments for model, runtime, and benchmark sweeps.
    """
    parser = argparse.ArgumentParser(
        description="Inference-only low-level profiler for Flux2 transformer (block, operator, and kernel views)."
    )
    parser.add_argument("--transformer-dir", type=Path, default=None, help="Directory containing config.json + weights.")
    parser.add_argument("--config", type=Path, default=Path("flux2_klein_transformer_config.json"))
    parser.add_argument("--weights", type=Path, default=None, help="Path to .safetensors/.bin weight file.")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-tokens", type=int, default=1024)
    parser.add_argument("--txt-tokens", type=int, default=256)

    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)

    parser.add_argument("--kernel-warmup", type=int, default=5)
    parser.add_argument("--kernel-iters", type=int, default=20)
    parser.add_argument("--sweep-seq-lens", type=str, default="256,512,1024")
    parser.add_argument("--sweep-heads", type=str, default="12,24")
    parser.add_argument("--sweep-hidden-dims", type=str, default="1536,3072")

    parser.add_argument("--output-json", type=Path, default=Path("flux2_profile_report.json"))
    return parser.parse_args()


def parse_dtype(dtype_str: str, device: str) -> torch.dtype:
    """
    Convert user dtype text into torch dtype.

    CPU fallback:
    - If fp16/bf16 is requested on CPU, return fp32 for compatibility and stability.
    """
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = mapping.get(dtype_str.lower())
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def load_json(path: Path) -> dict:
    """
    Load JSON file from disk and return dictionary content.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_weight_file(path_or_dir: Path | None) -> Path | None:
    """
    Resolve a weight file from a direct path or from a directory containing known filenames.
    """
    if path_or_dir is None:
        return None
    if path_or_dir.is_file():
        return path_or_dir
    for candidate in WEIGHT_CANDIDATES:
        candidate_path = path_or_dir / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def load_state_dict(weights_path: Path) -> dict[str, torch.Tensor]:
    """
    Load checkpoint and normalize it to a plain state_dict.

    Supported formats:
    - .safetensors file
    - torch .bin with direct tensor map
    - torch .bin with nested 'state_dict'
    """
    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(weights_path), device="cpu")

    payload = torch.load(weights_path, map_location="cpu")
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        tensor_values = all(torch.is_tensor(v) for v in payload.values())
        if tensor_values:
            return payload
    raise ValueError(f"Unsupported checkpoint payload format in {weights_path}")


def maybe_strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """
    Strip prefix from all keys if every key starts with that prefix.
    """
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def build_model(config_path: Path, weights_path: Path) -> Flux2Transformer2DModel:
    """
    Build Flux2 transformer from config and load checkpoint weights.

    Raises if required keys are missing after load.
    """
    config = load_json(config_path)
    model = Flux2Transformer2DModel.from_config(config)

    state_dict = load_state_dict(weights_path)
    # Common checkpoint namespaces to remove when present.
    for prefix in ("transformer.", "module.", "model."):
        state_dict = maybe_strip_prefix(state_dict, prefix)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        missing_preview = ", ".join(incompatible.missing_keys[:5])
        raise RuntimeError(
            f"Failed to load checkpoint cleanly. Missing {len(incompatible.missing_keys)} keys "
            f"(first keys: {missing_preview})"
        )
    return model


def build_token_ids(seq_len: int, axes_count: int, device: str) -> torch.Tensor:
    """
    Build deterministic token IDs used for RoPE indexing in Flux2.

    Output shape: [seq_len, axes_count]
    """
    base = torch.arange(seq_len, device=device, dtype=torch.long)
    cols = []
    stride = 1
    for _ in range(axes_count):
        cols.append((base // stride) % 1024)
        stride *= 1024
    return torch.stack(cols, dim=-1)


def count_params(module: torch.nn.Module) -> int:
    """
    Return total parameter count for a module.
    """
    return sum(p.numel() for p in module.parameters())


def param_bytes(module: torch.nn.Module) -> int:
    """
    Return parameter storage size in bytes for a module.
    """
    return sum(p.numel() * p.element_size() for p in module.parameters())


def maybe_sync(device: str) -> None:
    """
    Synchronize CUDA kernels before reading timers/memory counters.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def maybe_get_process():
    """
    Return psutil.Process when available, else None.
    Used for CPU RSS memory fallback metrics.
    """
    try:
        import psutil  # type: ignore

        return psutil.Process()
    except Exception:
        return None


@dataclass
class BlockRuntimeStats:
    """
    Accumulates per-block runtime and memory samples captured by forward hooks.
    """
    durations_ms: list[float]
    peak_memory_bytes: list[int]
    cpu_rss_delta_bytes: list[int]


class BlockProfiler:
    """
    Hook-based block profiler for Flux2 double/single stream transformer blocks.

    Per block:
    - pre-hook stores start timestamp and memory baseline
    - post-hook stores elapsed time and memory delta
    """

    def __init__(self, model: Flux2Transformer2DModel, device: str):
        self.model = model
        self.device = device
        self._hooks = []
        self._starts = {}
        self._cuda_mem_start = {}
        self._cpu_mem_start = {}
        self.stats: dict[str, BlockRuntimeStats] = {}
        self._process = maybe_get_process()

    def _ensure_block(self, name: str) -> None:
        """
        Initialize a stats bucket for a block name on first use.
        """
        if name not in self.stats:
            self.stats[name] = BlockRuntimeStats(durations_ms=[], peak_memory_bytes=[], cpu_rss_delta_bytes=[])

    def _pre_hook(self, name: str):
        """
        Create a pre-forward hook for one named block.
        """
        def hook(_module, _inputs):
            self._ensure_block(name)
            maybe_sync(self.device)
            self._starts[name] = time.perf_counter()
            if self.device.startswith("cuda") and torch.cuda.is_available():
                # Reset peak stats so the recorded peak corresponds to the current block call.
                torch.cuda.reset_peak_memory_stats(device=self.device)
                self._cuda_mem_start[name] = torch.cuda.memory_allocated(device=self.device)
            elif self._process is not None:
                self._cpu_mem_start[name] = int(self._process.memory_info().rss)

        return hook

    def _post_hook(self, name: str):
        """
        Create a post-forward hook for one named block.
        """
        def hook(_module, _inputs, _outputs):
            maybe_sync(self.device)
            t1 = time.perf_counter()
            dt_ms = (t1 - self._starts[name]) * 1e3
            self.stats[name].durations_ms.append(dt_ms)

            if self.device.startswith("cuda") and torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(device=self.device)
                start = self._cuda_mem_start.get(name, 0)
                self.stats[name].peak_memory_bytes.append(max(0, int(peak - start)))
            elif self._process is not None:
                rss_now = int(self._process.memory_info().rss)
                rss_start = self._cpu_mem_start.get(name, rss_now)
                self.stats[name].cpu_rss_delta_bytes.append(max(0, rss_now - rss_start))

        return hook

    def register(self) -> None:
        """
        Register hooks for all double and single stream transformer blocks.
        """
        for idx, block in enumerate(self.model.transformer_blocks):
            name = f"double_stream_block_{idx}"
            self._hooks.append(block.register_forward_pre_hook(self._pre_hook(name)))
            self._hooks.append(block.register_forward_hook(self._post_hook(name)))
        for idx, block in enumerate(self.model.single_transformer_blocks):
            name = f"single_stream_block_{idx}"
            self._hooks.append(block.register_forward_pre_hook(self._pre_hook(name)))
            self._hooks.append(block.register_forward_hook(self._post_hook(name)))

    def remove(self) -> None:
        """
        Remove all hooks to avoid duplicated measurements on subsequent calls.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def make_model_inputs(
    model: Flux2Transformer2DModel,
    batch_size: int,
    img_tokens: int,
    txt_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """
    Create synthetic but shape-correct Flux2 inputs for inference profiling.

    Returns a dict that can be passed directly into model(**inputs).
    """
    axes_count = len(model.config.axes_dims_rope)
    # Hidden/image stream tokens.
    hidden_states = torch.randn(batch_size, img_tokens, model.config.in_channels, device=device, dtype=dtype)
    # Text conditioning stream tokens.
    encoder_hidden_states = torch.randn(
        batch_size,
        txt_tokens,
        model.config.joint_attention_dim,
        device=device,
        dtype=dtype,
    )
    # Diffusion timestep per sample.
    timestep = torch.randint(low=1, high=1000, size=(batch_size,), device=device, dtype=torch.long)
    # Token IDs used by RoPE embedding logic.
    img_ids = build_token_ids(img_tokens, axes_count, device)
    txt_ids = build_token_ids(txt_tokens, axes_count, device)
    # Guidance tensor is only needed for guidance-distilled variants.
    guidance = (
        torch.randint(low=1, high=1000, size=(batch_size,), device=device, dtype=torch.long)
        if model.config.guidance_embeds
        else None
    )
    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "guidance": guidance,
        "return_dict": True,
    }


def run_block_profile(
    model: Flux2Transformer2DModel,
    model_inputs: dict,
    warmup: int,
    steps: int,
    device: str,
) -> tuple[list[dict], list[float]]:
    """
    Measure per-block latency/memory and full-step latency across repeated inference runs.
    """
    profiler = BlockProfiler(model, device)
    profiler.register()

    model.eval()
    with torch.no_grad():
        # Warmup stabilizes one-time kernel overheads and cache effects.
        for _ in range(warmup):
            _ = model(**model_inputs).sample
            maybe_sync(device)

    step_times_ms = []
    with torch.no_grad():
        for _ in range(steps):
            t0 = time.perf_counter()
            _ = model(**model_inputs).sample
            maybe_sync(device)
            t1 = time.perf_counter()
            step_times_ms.append((t1 - t0) * 1e3)

    profiler.remove()

    block_reports = []
    for name, stats in profiler.stats.items():
        avg_time = sum(stats.durations_ms) / max(1, len(stats.durations_ms))
        max_time = max(stats.durations_ms) if stats.durations_ms else 0.0
        if stats.peak_memory_bytes:
            avg_mem = int(sum(stats.peak_memory_bytes) / len(stats.peak_memory_bytes))
            max_mem = int(max(stats.peak_memory_bytes))
        elif stats.cpu_rss_delta_bytes:
            # CPU fallback path when CUDA counters are unavailable.
            avg_mem = int(sum(stats.cpu_rss_delta_bytes) / len(stats.cpu_rss_delta_bytes))
            max_mem = int(max(stats.cpu_rss_delta_bytes))
        else:
            avg_mem = 0
            max_mem = 0
        block_reports.append(
            {
                "name": name,
                "avg_time_ms": avg_time,
                "max_time_ms": max_time,
                "avg_memory_delta_bytes": avg_mem,
                "max_memory_delta_bytes": max_mem,
                "num_samples": len(stats.durations_ms),
            }
        )

    block_reports.sort(key=lambda x: x["name"])
    return block_reports, step_times_ms


def attach_block_param_stats(model: Flux2Transformer2DModel, block_reports: list[dict]) -> list[dict]:
    """
    Attach parameter count and parameter-bytes to each block report row.
    """
    param_map = {}
    for i, block in enumerate(model.transformer_blocks):
        param_map[f"double_stream_block_{i}"] = {
            "params": count_params(block),
            "param_bytes": param_bytes(block),
        }
    for i, block in enumerate(model.single_transformer_blocks):
        param_map[f"single_stream_block_{i}"] = {
            "params": count_params(block),
            "param_bytes": param_bytes(block),
        }

    out = []
    for row in block_reports:
        extra = param_map.get(row["name"], {"params": 0, "param_bytes": 0})
        out.append({**row, **extra})
    return out


def categorize_op(op_name: str) -> str | None:
    """
    Bucket raw profiler op names into high-level categories.
    """
    name = op_name.lower()
    if any(x in name for x in ("scaled_dot_product_attention", "flash_attention", "sdpa")):
        return "attention"
    if "softmax" in name:
        return "softmax"
    if any(x in name for x in ("layer_norm", "native_layer_norm", "rms_norm")):
        return "layernorm"
    if any(x in name for x in ("mm", "addmm", "bmm", "matmul", "gemm")):
        return "matmul_gemm"
    return None


def run_operator_profile(
    model: Flux2Transformer2DModel,
    model_inputs: dict,
    steps: int,
    device: str,
) -> tuple[list[dict], list[dict]]:
    """
    Profile full inference with torch.profiler and return:
    - grouped category metrics
    - raw per-op metrics
    """
    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    profile_kwargs = dict(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    )
    try:
        # with_flops is available in newer torch versions.
        prof_ctx = profile(with_flops=True, **profile_kwargs)
    except TypeError:
        # Fallback for older torch builds.
        prof_ctx = profile(**profile_kwargs)

    with prof_ctx as prof:
        with torch.no_grad():
            for _ in range(steps):
                _ = model(**model_inputs).sample
                prof.step()

    grouped = defaultdict(lambda: {"calls": 0, "self_time_us": 0.0, "total_time_us": 0.0, "flops": 0.0, "mem": 0})
    per_op_rows = []
    for evt in prof.key_averages():
        self_cuda_time = float(getattr(evt, "self_cuda_time_total", 0.0))
        self_cpu_time = float(getattr(evt, "self_cpu_time_total", 0.0))
        total_cuda_time = float(getattr(evt, "cuda_time_total", 0.0))
        total_cpu_time = float(getattr(evt, "cpu_time_total", 0.0))
        # Prefer CUDA timing fields when available, else use CPU timing.
        self_time = self_cuda_time if self_cuda_time > 0 else self_cpu_time
        total_time = total_cuda_time if total_cuda_time > 0 else total_cpu_time
        flops = float(getattr(evt, "flops", 0.0))
        mem = int(abs(getattr(evt, "self_cuda_memory_usage", 0)) + abs(getattr(evt, "self_cpu_memory_usage", 0)))

        category = categorize_op(evt.key)
        per_op_rows.append(
            {
                "op_name": evt.key,
                "category": category,
                "calls": int(evt.count),
                "self_time_ms": self_time / 1e3,
                "total_time_ms": total_time / 1e3,
                "self_memory_bytes": mem,
                "flops": flops,
            }
        )

        if category is None:
            # Keep uncategorized ops in raw rows, but skip grouped aggregation.
            continue
        grouped[category]["calls"] += int(evt.count)
        grouped[category]["self_time_us"] += self_time
        grouped[category]["total_time_us"] += total_time
        grouped[category]["flops"] += flops
        grouped[category]["mem"] += mem

    grouped_rows = []
    total_profile_us = sum(v["self_time_us"] for v in grouped.values())
    for category, row in grouped.items():
        # Percent share is computed from grouped self-time totals.
        pct = 100.0 * row["self_time_us"] / total_profile_us if total_profile_us > 0 else 0.0
        grouped_rows.append(
            {
                "category": category,
                "calls": row["calls"],
                "self_time_ms": row["self_time_us"] / 1e3,
                "total_time_ms": row["total_time_us"] / 1e3,
                "runtime_percent": pct,
                "flops": row["flops"],
                "self_memory_bytes": row["mem"],
            }
        )

    grouped_rows.sort(key=lambda x: x["self_time_ms"], reverse=True)
    per_op_rows.sort(key=lambda x: x["self_time_ms"], reverse=True)
    return grouped_rows, per_op_rows


def benchmark_kernel(
    fn: Callable[[], None],
    warmup: int,
    iters: int,
    device: str,
    process,
) -> tuple[float, int, int]:
    """
    Benchmark one callable workload and return measured latency/memory deltas.

    Returns:
    - avg_ms: average measured latency across iters
    - avg_mem: average memory delta across iters
    - max_mem: max memory delta across iters
    """
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        maybe_sync(device)

        durations = []
        mem_deltas = []
        for _ in range(iters):
            if device.startswith("cuda") and torch.cuda.is_available():
                # Reset and capture CUDA allocation baseline for this iteration.
                torch.cuda.reset_peak_memory_stats(device=device)
                start_mem = torch.cuda.memory_allocated(device=device)
            elif process is not None:
                start_mem = int(process.memory_info().rss)
            else:
                start_mem = 0

            t0 = time.perf_counter()
            fn()
            maybe_sync(device)
            t1 = time.perf_counter()
            durations.append((t1 - t0) * 1e3)

            if device.startswith("cuda") and torch.cuda.is_available():
                # Use peak allocated bytes observed during this single iteration.
                peak_mem = torch.cuda.max_memory_allocated(device=device)
                mem_deltas.append(max(0, int(peak_mem - start_mem)))
            elif process is not None:
                end_mem = int(process.memory_info().rss)
                mem_deltas.append(max(0, end_mem - start_mem))
            else:
                mem_deltas.append(0)

    avg_ms = sum(durations) / max(1, len(durations))
    avg_mem = int(sum(mem_deltas) / max(1, len(mem_deltas)))
    max_mem = int(max(mem_deltas)) if mem_deltas else 0
    return avg_ms, avg_mem, max_mem


def run_kernel_benchmarks(
    batch: int,
    seq_lens: list[int],
    heads_list: list[int],
    hidden_dims: list[int],
    dtype: torch.dtype,
    device: str,
    warmup: int,
    iters: int,
) -> list[dict]:
    """
    Run microbenchmarks for core transformer kernels over a shape sweep.

    Kernels:
    - GEMM (torch.matmul)
    - Scaled dot-product attention
    - Softmax
    - LayerNorm
    """
    rows = []
    process = maybe_get_process()

    for seq in seq_lens:
        for hidden_dim in hidden_dims:
            for heads in heads_list:
                # Invalid attention head split; skip this configuration.
                if hidden_dim % heads != 0:
                    continue
                head_dim = hidden_dim // heads

                # GEMM proxy for projection-heavy workloads.
                a = torch.randn(batch * seq, hidden_dim, device=device, dtype=dtype)
                b = torch.randn(hidden_dim, 3 * hidden_dim, device=device, dtype=dtype)
                gemm_fn = lambda: torch.matmul(a, b)
                gemm_ms, gemm_mem_avg, gemm_mem_max = benchmark_kernel(gemm_fn, warmup, iters, device, process)

                # SDPA benchmark using full [B, H, S, D] tensors.
                q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
                k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
                v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
                sdpa_fn = lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
                sdpa_ms, sdpa_mem_avg, sdpa_mem_max = benchmark_kernel(sdpa_fn, warmup, iters, device, process)

                # Softmax benchmark over attention score tensor.
                soft_in = torch.randn(batch, heads, seq, seq, device=device, dtype=dtype)
                softmax_fn = lambda: torch.softmax(soft_in, dim=-1)
                softmax_ms, softmax_mem_avg, softmax_mem_max = benchmark_kernel(softmax_fn, warmup, iters, device, process)

                # LayerNorm benchmark over hidden-state tensor.
                ln_in = torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype)
                ln_w = torch.ones(hidden_dim, device=device, dtype=dtype)
                ln_b = torch.zeros(hidden_dim, device=device, dtype=dtype)
                layernorm_fn = lambda: F.layer_norm(ln_in, (hidden_dim,), ln_w, ln_b, 1e-5)
                layernorm_ms, ln_mem_avg, ln_mem_max = benchmark_kernel(layernorm_fn, warmup, iters, device, process)

                for op_name, ms, avg_mem, max_mem in (
                    ("gemm", gemm_ms, gemm_mem_avg, gemm_mem_max),
                    ("sdpa", sdpa_ms, sdpa_mem_avg, sdpa_mem_max),
                    ("softmax", softmax_ms, softmax_mem_avg, softmax_mem_max),
                    ("layernorm", layernorm_ms, ln_mem_avg, ln_mem_max),
                ):
                    rows.append(
                        {
                            "op": op_name,
                            "batch": batch,
                            "seq_len": seq,
                            "heads": heads,
                            "hidden_dim": hidden_dim,
                            "head_dim": head_dim,
                            "measured_time_ms": ms,
                            "avg_memory_delta_bytes": avg_mem,
                            "max_memory_delta_bytes": max_mem,
                        }
                    )

    return rows


def print_top_blocks(block_rows: list[dict], limit: int = 12) -> None:
    """
    Console summary of block profiling rows sorted by avg latency.
    """
    print("\n=== Block Profile (top by measured avg time) ===")
    ranked = sorted(block_rows, key=lambda x: x["avg_time_ms"], reverse=True)
    for row in ranked[:limit]:
        print(
            f"{row['name']:<24} "
            f"time={row['avg_time_ms']:.3f} ms "
            f"mem={row['avg_memory_delta_bytes']/1e6:.2f} MB "
            f"params={row['params']/1e6:.2f}M"
        )


def print_operator_summary(grouped_rows: list[dict]) -> None:
    """
    Console summary of grouped operator categories.
    """
    print("\n=== Operator Profile (grouped) ===")
    if not grouped_rows:
        print("No grouped operator rows found.")
        return
    for row in grouped_rows:
        print(
            f"{row['category']:<12} "
            f"self_time={row['self_time_ms']:.3f} ms "
            f"runtime={row['runtime_percent']:.1f}% "
            f"calls={row['calls']} "
            f"mem={row['self_memory_bytes']/1e6:.2f} MB"
        )


def print_top_operators(per_op_rows: list[dict], limit: int = 20) -> None:
    """
    Console summary of top raw ops sorted by self time.
    """
    print("\n=== Operator Profile (top raw ops by self time) ===")
    for row in per_op_rows[:limit]:
        category = row["category"] if row["category"] is not None else "other"
        print(
            f"{category:<12} "
            f"{row['op_name']:<45} "
            f"time={row['self_time_ms']:.3f} ms "
            f"calls={row['calls']} "
            f"mem={row['self_memory_bytes']/1e6:.2f} MB"
        )


def print_kernel_summary(kernel_rows: list[dict], topk: int = 20) -> None:
    """
    Console summary of slowest sampled kernel benchmark rows.
    """
    print("\n=== Kernel Benchmarks (sample rows) ===")
    if not kernel_rows:
        print("No kernel rows produced.")
        return
    ranked = sorted(kernel_rows, key=lambda x: x["measured_time_ms"], reverse=True)
    for row in ranked[:topk]:
        print(
            f"{row['op']:<9} seq={row['seq_len']:<4} heads={row['heads']:<3} hidden={row['hidden_dim']:<4} "
            f"time={row['measured_time_ms']:.3f} ms "
            f"avg_mem={row['avg_memory_delta_bytes']/1e6:.2f} MB"
        )


def main() -> None:
    """
    Entry point:
    1) Parse args and resolve model files
    2) Build model and synthetic inputs
    3) Collect block/operator/kernel measurements
    4) Emit JSON report + concise console summaries
    """
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.transformer_dir is not None:
        config_path = args.transformer_dir / "config.json"
        weights_path = find_weight_file(args.transformer_dir)
    else:
        config_path = args.config
        weights_path = find_weight_file(args.weights)

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    if weights_path is None:
        raise FileNotFoundError("Could not locate weights. Pass --weights or --transformer-dir with model weights.")

    dtype = parse_dtype(args.dtype, args.device)
    model = build_model(config_path, weights_path).to(device=args.device, dtype=dtype).eval()
    # Build deterministic-shape random inputs for profiling.
    model_inputs = make_model_inputs(
        model=model,
        batch_size=args.batch_size,
        img_tokens=args.img_tokens,
        txt_tokens=args.txt_tokens,
        device=args.device,
        dtype=dtype,
    )

    block_reports, step_times_ms = run_block_profile(
        model=model,
        model_inputs=model_inputs,
        warmup=args.warmup,
        steps=args.steps,
        device=args.device,
    )
    block_reports = attach_block_param_stats(model, block_reports)

    op_grouped_rows, op_raw_rows = run_operator_profile(
        model=model,
        model_inputs=model_inputs,
        steps=max(2, args.steps),
        device=args.device,
    )

    kernel_rows = run_kernel_benchmarks(
        batch=args.batch_size,
        seq_lens=parse_csv_ints(args.sweep_seq_lens),
        heads_list=parse_csv_ints(args.sweep_heads),
        hidden_dims=parse_csv_ints(args.sweep_hidden_dims),
        dtype=dtype,
        device=args.device,
        warmup=args.kernel_warmup,
        iters=args.kernel_iters,
    )

    total_params = count_params(model)
    avg_step_ms = sum(step_times_ms) / max(1, len(step_times_ms))
    max_step_ms = max(step_times_ms) if step_times_ms else 0.0

    report = {
        # Metadata used to reproduce the profiling run.
        "meta": {
            "config_path": str(config_path.resolve()),
            "weights_path": str(weights_path.resolve()),
            "device": args.device,
            "dtype": str(dtype),
            "batch_size": args.batch_size,
            "img_tokens": args.img_tokens,
            "txt_tokens": args.txt_tokens,
            "warmup": args.warmup,
            "steps": args.steps,
        },
        "model_summary": {
            "total_params": total_params,
            "total_params_billion": total_params / 1e9,
            "avg_step_time_ms": avg_step_ms,
            "max_step_time_ms": max_step_ms,
        },
        # Detailed measured sections.
        "block_profile": block_reports,
        "operator_profile_grouped": op_grouped_rows,
        "operator_profile_raw": op_raw_rows,
        "kernel_benchmarks": kernel_rows,
    }

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Loaded config: {config_path}")
    print(f"Loaded weights: {weights_path}")
    print(f"Total params: {total_params / 1e9:.3f}B")
    print(f"Avg full forward latency: {avg_step_ms:.3f} ms")
    print(f"Profile report written to: {args.output_json}")

    print_top_blocks(block_reports)
    print_operator_summary(op_grouped_rows)
    print_top_operators(op_raw_rows)
    print_kernel_summary(kernel_rows)


if __name__ == "__main__":
    main()
