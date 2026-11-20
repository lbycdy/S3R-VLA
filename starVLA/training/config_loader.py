"""Training-config loading with inheritance (imports only omegaconf, so the launcher can read a config cheaply)."""
import os
import sys

from omegaconf import OmegaConf


def load_config(path: str):
    """Load a YAML config. A top-level `_base_: <file>` (relative to this config) is loaded first and then overridden
    by this config; the merged config is what the trainer saves to <output_dir>/config.yaml."""
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = OmegaConf.merge(load_config(os.path.join(os.path.dirname(os.path.abspath(path)), base)), cfg)
    return cfg


if __name__ == "__main__":
    # python -m starVLA.training.config_loader <config.yaml> <key> [<key> ...]: print the values (used by scripts/train.sh)
    cfg = load_config(sys.argv[1])
    print(" ".join(str(OmegaConf.select(cfg, k, default="")) for k in sys.argv[2:]))
