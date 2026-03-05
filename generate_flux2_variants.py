"""
Generate smaller Flux2 transformer variants from a larger base checkpoint.

This script changes depth-related config fields (`num_layers`, `num_single_layers`)
to approximate requested target model sizes (for example 4B -> 2B / 1B), then
saves variant checkpoints with compatible transformer weights.
"""

import argparse
import copy
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
    """
    Parse command-line arguments for base model loading and target variant generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate smaller Flux2 transformer variants (for example 4B -> 2B/1B) by reducing layer counts."
    )
    parser.add_argument(
        "--base-transformer-dir",
        type=Path,
        default=None,
        help="Directory that contains config.json + transformer weights.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("flux2_klein_transformer_config.json"),
        help="Path to base transformer config JSON.",
    )
    parser.add_argument(
        "--base-weights",
        type=Path,
        default=None,
        help="Path to base transformer weight file (.safetensors/.bin).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("flux2_variants"),
        help="Where generated variant folders will be written.",
    )
    parser.add_argument(
        "--base-size-b",
        type=float,
        default=4.0,
        help="Declared size (in billions) for the base model. Used to map targets to ratios.",
    )
    parser.add_argument(
        "--targets-b",
        type=float,
        nargs="+",
        default=[2.0, 1.0],
        help="Target sizes in billions. Example: --targets-b 2 1",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save generated models as safetensors.",
    )
    parser.add_argument(
        "--allow-random-base",
        action="store_true",
        help="If weights are not provided/found, use random base initialization.",
    )
    return parser.parse_args()


def count_params(module: torch.nn.Module) -> int:
    """
    Return total number of parameters for a module.
    """
    return sum(p.numel() for p in module.parameters())


def load_json(path: Path) -> dict:
    """
    Load JSON file from disk.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_weight_file(path_or_dir: Path | None) -> Path | None:
    """
    Resolve weight file path from a direct path or candidate filenames in a directory.
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
    Load checkpoint and normalize to a plain tensor state dict.
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
    Strip a key prefix if all keys are namespaced under it.
    """
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def build_base_model(config_path: Path, weights_path: Path | None, allow_random_base: bool):
    """
    Build base model from config and optionally load base weights.

    Returns:
    - model
    - config dict
    - bool indicating whether external base weights were loaded
    """
    config = load_json(config_path)
    model = Flux2Transformer2DModel.from_config(config)

    if weights_path is None:
        if not allow_random_base:
            raise FileNotFoundError(
                "No base weights found. Pass --base-weights/--base-transformer-dir or set --allow-random-base."
            )
        return model, config, False

    state_dict = load_state_dict(weights_path)

    # Common checkpoints may store weights with one extra namespace level.
    for prefix in ("transformer.", "module.", "model."):
        state_dict = maybe_strip_prefix(state_dict, prefix)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        missing_preview = ", ".join(incompatible.missing_keys[:5])
        raise RuntimeError(
            f"Failed to load base checkpoint cleanly. Missing {len(incompatible.missing_keys)} keys "
            f"(first keys: {missing_preview})"
        )
    return model, config, True


def pick_layer_counts_for_ratio(base_model: Flux2Transformer2DModel, target_ratio: float) -> tuple[int, int, int]:
    """
    Search over possible layer-count combinations and pick the best depth pair.

    The selected combination minimizes:
    - parameter-count distance to target ratio
    - imbalance between double/single stream depth shrinkage ratios
    """
    base_double = len(base_model.transformer_blocks)
    base_single = len(base_model.single_transformer_blocks)
    if base_double == 0 or base_single == 0:
        raise ValueError("Base model must have both transformer block stacks.")

    total = count_params(base_model)
    per_double = count_params(base_model.transformer_blocks[0])
    per_single = count_params(base_model.single_transformer_blocks[0])
    shared = total - (base_double * per_double + base_single * per_single)
    target_total = int(total * target_ratio)

    best = None
    for n_double in range(1, base_double + 1):
        for n_single in range(1, base_single + 1):
            total_candidate = shared + n_double * per_double + n_single * per_single
            size_err = abs(total_candidate - target_total)
            depth_ratio_err = abs((n_double / base_double) - (n_single / base_single))
            score = size_err + 0.01 * total * depth_ratio_err
            if best is None or score < best[0]:
                best = (score, n_double, n_single, total_candidate)

    _, chosen_double, chosen_single, chosen_total = best
    return chosen_double, chosen_single, chosen_total


def format_size_tag(size_b: float) -> str:
    """
    Convert size float to folder-safe tag string (for example 2.0 -> '2b').
    """
    if size_b.is_integer():
        return f"{int(size_b)}b"
    return f"{size_b:.2f}".rstrip("0").rstrip(".") + "b"


def main() -> None:
    """
    End-to-end variant generation workflow:
    1) Load base model/config
    2) Pick target layer counts for each requested size
    3) Build each variant and load compatible weights
    4) Save variant checkpoint and manifest
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.base_transformer_dir is not None:
        config_path = args.base_transformer_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find config.json in {args.base_transformer_dir}")
        weights_path = find_weight_file(args.base_transformer_dir)
    else:
        config_path = args.base_config
        weights_path = find_weight_file(args.base_weights)

    base_model, base_config, loaded_weights = build_base_model(
        config_path=config_path,
        weights_path=weights_path,
        allow_random_base=args.allow_random_base,
    )
    base_state = base_model.state_dict()
    base_params = count_params(base_model)

    print(f"Base params: {base_params / 1e9:.3f}B")
    print(f"Base weights loaded: {loaded_weights}")

    manifest = {
        "base_config": str(config_path.resolve()),
        "base_weights": str(weights_path.resolve()) if weights_path is not None else None,
        "base_params": base_params,
        "base_size_b_declared": args.base_size_b,
        "variants": [],
    }

    for target_b in args.targets_b:
        ratio = target_b / args.base_size_b
        if ratio <= 0:
            print(f"Skipping invalid target {target_b}B")
            continue

        target_double, target_single, predicted_params = pick_layer_counts_for_ratio(base_model, ratio)
        variant_config = copy.deepcopy(base_config)
        variant_config["num_layers"] = target_double
        variant_config["num_single_layers"] = target_single

        variant_model = Flux2Transformer2DModel.from_config(variant_config)
        incompatible = variant_model.load_state_dict(base_state, strict=False)
        if incompatible.missing_keys:
            missing_preview = ", ".join(incompatible.missing_keys[:5])
            raise RuntimeError(
                f"Variant {target_b}B load failed with {len(incompatible.missing_keys)} missing keys. "
                f"First keys: {missing_preview}"
            )

        variant_tag = format_size_tag(target_b)
        variant_dir = args.output_dir / f"flux2_{variant_tag}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_model.save_pretrained(str(variant_dir), safe_serialization=args.safe_serialization)

        actual_params = count_params(variant_model)
        info = {
            "target_b": target_b,
            "variant_dir": str(variant_dir.resolve()),
            "num_layers": target_double,
            "num_single_layers": target_single,
            "predicted_params": predicted_params,
            "actual_params": actual_params,
            "actual_size_b": actual_params / 1e9,
        }
        manifest["variants"].append(info)
        print(
            f"{variant_tag}: num_layers={target_double}, num_single_layers={target_single}, "
            f"params={actual_params / 1e9:.3f}B"
        )

    manifest_path = args.output_dir / "variant_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
