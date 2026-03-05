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
