import argparse
import json
import re
from pathlib import Path

import torch

try:
    from safetensors.torch import load_file as load_safetensors_file
    from safetensors.torch import save_file as save_safetensors_file
except ImportError:  # pragma: no cover
    load_safetensors_file = None
    save_safetensors_file = None


WEIGHT_CANDIDATES = [
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "model.safetensors",
    "model.bin",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a pruned Flux2 transformer checkpoint by removing selected "
            "single transformer blocks using 1-based indexing."
        )
    )
    parser.add_argument(
        "--base-transformer-dir",
        type=Path,
        help="Transformer directory containing config.json and a checkpoint file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Explicit input checkpoint path (.safetensors or .bin/.pt/.pth).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Input Flux2 config JSON path.",
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        required=True,
        help="Path for the rewritten checkpoint.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        help="Optional output config path. Defaults to <output-weights parent>/config.json.",
    )
    parser.add_argument(
        "--skip-single-blocks",
        type=int,
        nargs="+",
        default=[13, 14, 15, 16],
        help="1-based single transformer block indices to remove. Default: 13 14 15 16.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def find_weight_file(path_or_dir: Path):
    if path_or_dir.is_file():
        return path_or_dir

    for name in WEIGHT_CANDIDATES:
        candidate = path_or_dir / name
        if candidate.exists():
            return candidate

    return None


def load_state_dict(weights_path: Path):
    suffix = weights_path.suffix.lower()
    if suffix == ".safetensors":
        if load_safetensors_file is None:
            raise ImportError("safetensors is required to read .safetensors checkpoints.")
        return load_safetensors_file(str(weights_path))

    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint structure in {weights_path}.")
    return checkpoint


def save_state_dict(weights_path: Path, state_dict):
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = weights_path.suffix.lower()
    if suffix == ".safetensors":
        if save_safetensors_file is None:
            raise ImportError("safetensors is required to write .safetensors checkpoints.")
        save_safetensors_file(state_dict, str(weights_path))
        return
    torch.save(state_dict, weights_path)


def prune_single_transformer_blocks(state_dict, skip_blocks_1_based):
    skip_zero_based = {index - 1 for index in skip_blocks_1_based}
    pattern = re.compile(r"^(?P<prefix>.*single_transformer_blocks\.)(?P<index>\d+)(?P<suffix>\..+)$")

    remapped_state_dict = {}
    removed_keys = []

    for key, value in state_dict.items():
        match = pattern.match(key)
        if not match:
            remapped_state_dict[key] = value
            continue

        block_index = int(match.group("index"))
        if block_index in skip_zero_based:
            removed_keys.append(key)
            continue

        new_index = block_index - sum(skipped < block_index for skipped in skip_zero_based)
        new_key = f"{match.group('prefix')}{new_index}{match.group('suffix')}"
        remapped_state_dict[new_key] = value

    return remapped_state_dict, removed_keys


def main():
    args = parse_args()

    if (args.base_transformer_dir is None) == (args.weights is None):
        raise ValueError("Provide exactly one of --base-transformer-dir or --weights.")

    weights_path = args.weights
    if args.base_transformer_dir is not None:
        weights_path = find_weight_file(args.base_transformer_dir)
        if weights_path is None:
            raise FileNotFoundError(
                f"Could not find a checkpoint in {args.base_transformer_dir}. "
                f"Tried: {', '.join(WEIGHT_CANDIDATES)}"
            )

    config = load_json(args.config)
    original_single_layers = config.get("num_single_layers")
    if original_single_layers is None:
        raise KeyError("Config is missing 'num_single_layers'.")

    skip_blocks = sorted(set(args.skip_single_blocks))
    if any(index < 1 or index > original_single_layers for index in skip_blocks):
        raise ValueError(
            f"Skip indices must be within 1..{original_single_layers}. Got: {skip_blocks}"
        )

    state_dict = load_state_dict(weights_path)
    pruned_state_dict, removed_keys = prune_single_transformer_blocks(state_dict, skip_blocks)

    expected_single_layers = original_single_layers - len(skip_blocks)
    config["num_single_layers"] = expected_single_layers

    output_config = args.output_config
    if output_config is None:
        output_config = args.output_weights.parent / "config.json"

    save_state_dict(args.output_weights, pruned_state_dict)
    save_json(output_config, config)

    print(f"Input checkpoint: {weights_path}")
    print(f"Output checkpoint: {args.output_weights}")
    print(f"Output config: {output_config}")
    print(f"Removed single blocks (1-based): {skip_blocks}")
    print(f"Removed parameter tensors: {len(removed_keys)}")
    print(f"num_single_layers: {original_single_layers} -> {expected_single_layers}")


if __name__ == "__main__":
    main()
