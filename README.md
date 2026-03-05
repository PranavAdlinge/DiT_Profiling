# DiT Profiling

Profiles Diffusion Transformers (DiTs) to compute parameter counts and GMACs across model variants and resolutions. Includes scripts to measure total/trainable parameters and forward-pass compute for diffusion-based transformer architectures.

---

## Features

- Total parameter count
- Trainable parameter count
- GMACs (Giga Multiply–Accumulate operations)
- Resolution-aware profiling
- Batch-size configurable
- Works with custom DiT implementations

---

## Installation

```bash
git clone https://github.com/yourusername/dit-profiling.git
cd dit-profiling
pip install -r requirements.txt
```

## Generate 4B -> 2B/1B Transformer Variants

```bash
python generate_flux2_variants.py \
  --base-transformer-dir /path/to/4b/transformer \
  --base-size-b 4 \
  --targets-b 2 1 \
  --output-dir ./flux2_variants \
  --safe-serialization
```

This creates variant folders like `flux2_2b` and `flux2_1b`, each containing:
- `config.json` with updated `num_layers` / `num_single_layers`
- transformer weight file copied from the base checkpoint where shapes match

## Run Dummy Inference On A Variant

```bash
python infer_flux2_variant.py --transformer-dir ./flux2_variants/flux2_2b
```

## Detailed Walkthrough: `generate_flux2_variants.py`

### What it does

- Loads a base Flux2 transformer config + checkpoint.
- Computes smaller depth configurations that best match requested target sizes.
- Builds variant models with updated `num_layers` and `num_single_layers`.
- Copies all shape-compatible transformer weights into each variant.
- Saves each variant folder and a `variant_manifest.json`.

### Function map

| Function | What it expects | What it does | Output |
|---|---|---|---|
| `parse_args` | CLI flags | Parses base model path, target sizes, output settings | Namespace |
| `count_params` | Module | Counts parameters | Integer |
| `load_json` | Path | Reads config JSON | Dict |
| `find_weight_file` | File/dir path | Resolves checkpoint file from candidates | Path or None |
| `load_state_dict` | Weight file path | Loads safetensors/bin and returns state dict | Tensor dict |
| `maybe_strip_prefix` | state dict + prefix | Removes common namespace prefixes | Updated dict |
| `build_base_model` | config path + optional weight path | Builds base model and loads weights if available | `(model, config, loaded_weights_bool)` |
| `pick_layer_counts_for_ratio` | Base model + target ratio | Searches best double/single depth pair to match target size | `(num_layers, num_single_layers, predicted_params)` |
| `format_size_tag` | Float size | Converts to tag for folder naming | String like `2b` |
| `main` | CLI entrypoint | Runs full variant generation pipeline | Saved variants + manifest |

## Detailed Walkthrough: `infer_flux2_variant.py`

### What it does

- Loads one Flux2 transformer checkpoint (base or generated variant).
- Creates synthetic inputs matching model config.
- Runs one no-grad forward pass.
- Prints model size and output tensor shape.

### Function map

| Function | What it expects | What it does | Output |
|---|---|---|---|
| `parse_args` | CLI flags | Parses model path, runtime, and synthetic input sizes | Namespace |
| `load_json` | Path | Reads config JSON | Dict |
| `find_weight_file` | File/dir path | Resolves checkpoint file from known names | Path or None |
| `load_state_dict` | Weight file path | Loads state dict from safetensors/bin | Tensor dict |
| `maybe_strip_prefix` | state dict + prefix | Removes common namespaces | Updated dict |
| `parse_dtype` | Dtype text + device | Maps dtype with CPU fallback behavior | `torch.dtype` |
| `build_token_ids` | seq len + axes + device | Generates deterministic RoPE token IDs | Tensor |
| `build_model` | Config path + weight path | Constructs model and loads weights | Model |
| `main` | CLI entrypoint | Runs full validation inference flow | Console output |

## Low-Level Transformer Profiling

```bash
python profile_flux2.py \
  --transformer-dir /path/to/4b/transformer \
  --batch-size 1 \
  --img-tokens 1024 \
  --txt-tokens 256 \
  --warmup 3 \
  --steps 10 \
  --sweep-seq-lens 256,512,1024 \
  --sweep-heads 12,24 \
  --sweep-hidden-dims 1536,3072 \
  --output-json flux2_profile_report.json
```

The report includes:
- block-level latency, memory delta, and parameter counts
- operator-level runtime and memory from `torch.profiler` (both grouped and raw op rows)
- kernel microbenchmarks for `GEMM`, `SDPA`, `softmax`, and `layernorm` with measured latency + memory deltas

## Detailed Walkthrough: `profile_flux2.py`

This section explains exactly what each code block/function does, what inputs it expects, and what output it produces.

### 1) High-Level Pipeline

`profile_flux2.py` executes in this order:

1. Parse CLI args.
2. Resolve model config + weight file.
3. Load the Flux2 transformer.
4. Build synthetic but shape-correct inputs.
5. Run block-level profiling with hooks.
6. Run operator-level profiling with `torch.profiler`.
7. Run kernel-level microbenchmarks (GEMM/SDPA/softmax/layernorm).
8. Save all measured metrics to JSON.
9. Print readable summaries.

Important: this script intentionally uses measured inference data only. It does not use hardcoded formula estimates for block latency.

### 2) Model/File Inputs

You can load the model in two ways:

- Directory mode:
  - `--transformer-dir /path/to/transformer_dir`
  - Expected files:
    - `config.json`
    - one of:
      - `diffusion_pytorch_model.safetensors`
      - `diffusion_pytorch_model.bin`
      - `pytorch_model.bin`
      - `model.safetensors`
      - `model.bin`

- Explicit file mode:
  - `--config /path/to/config.json`
  - `--weights /path/to/weights.safetensors`

### 3) Function-by-Function Guide (`profile_flux2.py`)

| Function / Class | Inputs | What it does | Output |
|---|---|---|---|
| `parse_csv_ints(csv_str)` | Comma-separated string like `"256,512"` | Splits, trims, converts to integers | `list[int]` |
| `parse_args()` | CLI args | Declares all script flags | Parsed `argparse.Namespace` |
| `parse_dtype(dtype_str, device)` | Dtype text + device | Maps text to torch dtype; CPU fallback to fp32 for fp16/bf16 requests | `torch.dtype` |
| `load_json(path)` | JSON file path | Reads config JSON | `dict` |
| `find_weight_file(path_or_dir)` | File or directory path | Resolves weight path from known candidate names | `Path` or `None` |
| `load_state_dict(weights_path)` | Weight file path | Loads `.safetensors` or `.bin`, normalizes to state dict | `dict[str, Tensor]` |
| `maybe_strip_prefix(state_dict, prefix)` | state dict + key prefix | Removes uniform namespace prefixes (`transformer.`, `module.`, etc.) | Updated state dict |
| `build_model(config_path, weights_path)` | Config + weights | Builds `Flux2Transformer2DModel` and loads weights | Model instance |
| `build_token_ids(seq_len, axes_count, device)` | Sequence length, rope axes count | Creates deterministic RoPE token IDs | Tensor `[seq_len, axes_count]` |
| `count_params(module)` | Torch module | Counts total params | `int` |
| `param_bytes(module)` | Torch module | Calculates bytes occupied by module params | `int` |
| `maybe_sync(device)` | Device string | Calls `torch.cuda.synchronize()` on CUDA to make timing accurate | `None` |
| `maybe_get_process()` | None | Creates `psutil.Process()` when available for CPU RSS fallback | `Process` or `None` |
| `BlockRuntimeStats` | N/A | Dataclass container for per-block timing/memory samples | Dataclass object |
| `BlockProfiler` | Model + device | Registers per-block pre/post hooks, records time and memory deltas | Internal stats in `self.stats` |
| `make_model_inputs(...)` | Model + batch/token/device/dtype | Creates shape-correct random hidden/text/timestep/id/guidance inputs | Model input dict |
| `run_block_profile(...)` | Model + inputs + warmup/steps | Runs full forward passes; captures per-block and full-step timing | `(block_rows, step_times_ms)` |
| `attach_block_param_stats(model, block_reports)` | Model + block rows | Adds `params` and `param_bytes` per block | Updated block rows |
| `categorize_op(op_name)` | Raw op name string | Buckets ops into `attention`, `softmax`, `layernorm`, `matmul_gemm` | Category string or `None` |
| `run_operator_profile(...)` | Model + inputs + steps | Uses `torch.profiler`, returns grouped and raw op stats | `(grouped_rows, raw_rows)` |
| `benchmark_kernel(...)` | Callable kernel + warmup/iters | Benchmarks one kernel call repeatedly and records time/memory deltas | `(avg_ms, avg_mem_bytes, max_mem_bytes)` |
| `run_kernel_benchmarks(...)` | Sweep dimensions | Benchmarks GEMM/SDPA/softmax/layernorm over shape combinations | List of benchmark rows |
| `print_top_blocks(...)` | Block rows | Console summary of slowest blocks | Printed text |
| `print_operator_summary(...)` | Grouped op rows | Console summary by grouped op category | Printed text |
| `print_top_operators(...)` | Raw op rows | Console summary of top individual ops | Printed text |
| `print_kernel_summary(...)` | Kernel benchmark rows | Console summary of slowest sampled kernel configs | Printed text |
| `main()` | CLI entrypoint | Orchestrates complete profiling workflow and writes JSON report | JSON file + stdout summary |

### 4) What `run_block_profile` Is Actually Measuring

- Hooks are attached to:
  - every `double_stream_block_i`
  - every `single_stream_block_i`
- For each block invocation:
  - pre-hook captures start time and memory baseline
  - post-hook captures end time and memory delta
- Memory metric:
  - CUDA path: uses `max_memory_allocated - memory_allocated_start`
  - CPU fallback path: uses process RSS delta via `psutil`
- Full step timing is measured separately around the full `model(**inputs)` call.

### 5) What `run_operator_profile` Is Actually Measuring

`torch.profiler` captures runtime events from real model execution.

Per raw op row:
- `op_name`
- `calls`
- `self_time_ms`
- `total_time_ms`
- `self_memory_bytes`
- `flops` (if profiler runtime supports FLOP reporting)
- `category` (derived by `categorize_op`)

Grouped output merges only categorized ops and adds:
- `runtime_percent` based on grouped self-time totals.

### 6) What Kernel Benchmark Sweep Measures

For each valid `(seq_len, heads, hidden_dim)` combination:

- GEMM:
  - runs `torch.matmul` on projection-like matrices
- SDPA:
  - runs `torch.nn.functional.scaled_dot_product_attention`
- Softmax:
  - runs `torch.softmax` over `[B, H, S, S]` tensor
- LayerNorm:
  - runs `torch.nn.functional.layer_norm` over `[B, S, D]`

For each op/config row, recorded fields are:
- `measured_time_ms`
- `avg_memory_delta_bytes`
- `max_memory_delta_bytes`

### 7) JSON Report Schema

`flux2_profile_report.json` contains:

- `meta`
  - model paths, runtime device/dtype, batch/tokens, warmup/steps
- `model_summary`
  - `total_params`
  - `total_params_billion`
  - `avg_step_time_ms`
  - `max_step_time_ms`
- `block_profile`
  - one row per block with latency/memory + params
- `operator_profile_grouped`
  - grouped categories from profiler
- `operator_profile_raw`
  - raw profiler key averages
- `kernel_benchmarks`
  - microbenchmark rows over sweep shapes

### 8) Practical Notes

- If CUDA is available, timing values are synchronized before measurement to avoid async under-reporting.
- If `psutil` is unavailable, CPU memory fallback rows may have zeros for memory deltas.
- `warmup` and `steps` strongly affect stability; use larger values for final measurements.
- For mobile/on-device studies, run with dimensions that match your expected deployment sequence lengths.
