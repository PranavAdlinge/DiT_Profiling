import argparse
import json
from pathlib import Path

import torch

from transformer_flux2 import Flux2Transformer2DModel


WEIGHT_CANDIDATES = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "model.safetensors",
    "model.bin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dummy inference on a Flux2 transformer checkpoint/variant.")
    parser.add_argument(
        "--transformer-dir",
        type=Path,
        default=None,
        help="Directory with config.json + weights.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("flux2_klein_transformer_config.json"),
        help="Path to config JSON (used if --transformer-dir is not provided).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to weight file (used if --transformer-dir is not provided).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-tokens", type=int, default=1024)
    parser.add_argument("--txt-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_weight_file(path_or_dir: Path | None) -> Path | None:
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
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def parse_dtype(dtype_str: str, device: str) -> torch.dtype:
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


def build_token_ids(seq_len: int, axes_count: int, device: str) -> torch.Tensor:
    # Build deterministic token IDs across all RoPE axes.
    base = torch.arange(seq_len, device=device, dtype=torch.long)
    cols = []
    stride = 1
    for _ in range(axes_count):
        cols.append((base // stride) % 1024)
        stride *= 1024
    return torch.stack(cols, dim=-1)


def build_model(config_path: Path, weights_path: Path) -> Flux2Transformer2DModel:
    config = load_json(config_path)
    model = Flux2Transformer2DModel.from_config(config)

    state_dict = load_state_dict(weights_path)
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


def main() -> None:
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

    model = build_model(config_path, weights_path)
    dtype = parse_dtype(args.dtype, args.device)
    model = model.to(device=args.device, dtype=dtype).eval()

    axes_count = len(model.config.axes_dims_rope)

    hidden_states = torch.randn(
        args.batch_size,
        args.img_tokens,
        model.config.in_channels,
        device=args.device,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        args.batch_size,
        args.txt_tokens,
        model.config.joint_attention_dim,
        device=args.device,
        dtype=dtype,
    )
    timestep = torch.randint(low=1, high=1000, size=(args.batch_size,), device=args.device, dtype=torch.long)
    img_ids = build_token_ids(args.img_tokens, axes_count, args.device)
    txt_ids = build_token_ids(args.txt_tokens, axes_count, args.device)
    guidance = (
        torch.randint(low=1, high=1000, size=(args.batch_size,), device=args.device, dtype=torch.long)
        if model.config.guidance_embeds
        else None
    )

    with torch.no_grad():
        out = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=True,
        ).sample

    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {config_path}")
    print(f"Weights: {weights_path}")
    print(f"Device/DType: {args.device}/{dtype}")
    print(f"Params: {params / 1e9:.3f}B")
    print(f"Output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
