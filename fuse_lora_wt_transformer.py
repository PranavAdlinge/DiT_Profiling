import argparse
import inspect
from collections.abc import Mapping
from pathlib import Path

import torch
from peft import LoraConfig

from config import load_config
from transformer_flux2 import Flux2Transformer2DModel


LORA_WEIGHT_CANDIDATES = [
    "pytrch_model_fsdp.bin",
    "pytorch_model_fsdp.bin",
]

LORA_KEY_MARKERS = (
    ".lora_",
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a Flux2 transformer and LoRA checkpoint, fuse the LoRA into the base "
            "weights, and save a checkpoint without separate LoRA tensors."
        )
    )
    parser.add_argument("--dit-pretrained-path", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output-weights", type=Path, required=True)
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Scale to use while fusing the LoRA into the base weights.",
    )
    return parser.parse_args()


def resolve_lora_checkpoint(lora_path: Path) -> Path:
    if lora_path.is_file():
        return lora_path

    for name in LORA_WEIGHT_CANDIDATES:
        candidate = lora_path / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find a LoRA checkpoint in {lora_path}. "
        f"Tried: {', '.join(LORA_WEIGHT_CANDIDATES)}"
    )


def normalize_target_modules(target_modules):
    if target_modules is None:
        return None
    if isinstance(target_modules, str):
        modules = target_modules.split(",")
    else:
        modules = list(target_modules)
    return tuple(module.strip() for module in modules if str(module).strip())


def config_to_dict(config_obj):
    if isinstance(config_obj, Mapping):
        return dict(config_obj)

    if hasattr(config_obj, "__dict__"):
        return dict(vars(config_obj))

    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(config_obj, resolve=True)
    except ImportError:
        pass

    raise TypeError("Unsupported LoRA config object; expected a mapping-like or attribute-based config.")


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise TypeError("Unsupported checkpoint structure for LoRA weights.")


def fuse_and_unload_lora(transformer, lora_scale):
    if not hasattr(transformer, "fuse_lora"):
        raise AttributeError("Flux2Transformer2DModel does not expose fuse_lora().")

    fuse_lora_sig = inspect.signature(transformer.fuse_lora)
    fuse_kwargs = {}
    if "lora_scale" in fuse_lora_sig.parameters:
        fuse_kwargs["lora_scale"] = lora_scale
    if "components" in fuse_lora_sig.parameters:
        fuse_kwargs["components"] = ["transformer"]
    transformer.fuse_lora(**fuse_kwargs)

    if hasattr(transformer, "unload_lora"):
        transformer.unload_lora()
        return

    if hasattr(transformer, "unload_lora_weights"):
        transformer.unload_lora_weights(reset_to_overwritten_params=False)
        return

    if hasattr(transformer, "delete_adapters"):
        active_adapters = []
        if hasattr(transformer, "active_adapters"):
            adapters_attr = transformer.active_adapters
            active_adapters = list(adapters_attr() if callable(adapters_attr) else adapters_attr)
        if active_adapters:
            transformer.delete_adapters(active_adapters)


def strip_lora_keys(state_dict):
    return {
        key: value
        for key, value in state_dict.items()
        if not any(marker in key for marker in LORA_KEY_MARKERS)
    }


def main():
    args = parse_args()

    cfg = load_config(args.config_path)
    lora_cfg = getattr(cfg, "lora", None)
    if lora_cfg is None:
        raise ValueError(f"No 'lora' config found in {args.config_path}.")

    transformer = Flux2Transformer2DModel.from_pretrained(args.dit_pretrained_path)

    lora_cfg_dict = config_to_dict(lora_cfg)
    lora_cfg_dict["target_modules"] = normalize_target_modules(lora_cfg_dict.get("target_modules"))
    lora_config = LoraConfig(**lora_cfg_dict)
    transformer.add_adapter(lora_config)

    lora_weights_path = resolve_lora_checkpoint(args.lora_path)
    lora_state_dict = unwrap_state_dict(torch.load(lora_weights_path, map_location="cpu"))
    missing_keys, unexpected_keys = transformer.load_state_dict(lora_state_dict, strict=False)

    fuse_and_unload_lora(transformer, args.lora_scale)
    fused_state_dict = strip_lora_keys(transformer.state_dict())

    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fused_state_dict, args.output_weights)

    print(f"Base transformer: {args.dit_pretrained_path}")
    print(f"LoRA checkpoint: {lora_weights_path}")
    print(f"Output checkpoint: {args.output_weights}")
    print(f"Missing keys while loading LoRA: {len(missing_keys)}")
    print(f"Unexpected keys while loading LoRA: {len(unexpected_keys)}")
    print(f"Saved tensors after LoRA-key stripping: {len(fused_state_dict)}")


if __name__ == "__main__":
    main()
