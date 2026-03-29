"""Compute and visualise property distributions over the training dataset.

Uses the hydra config to instantiate the exact same train dataset as used in
training (via cfg.data.preprocessing.train_dataset).  Raw Data objects are
accessed through dataset.get(i) for PyG datasets and dataset[i] otherwise,
bypassing any FlowMatching transforms, so that .z and .y are always available.

Outputs (saved next to the dataset's processed directory by default):
  <out_prefix>_properties.npz   – raw collected arrays, one key per property
  <out_prefix>_distributions.png – one histogram subplot per property

Usage:
  python scripts/compute_property_distributions.py data=qm9
  python scripts/compute_property_distributions.py data=geom
  python scripts/compute_property_distributions.py data=qm9 out_prefix=results/qm9_train
"""

import math
import os

import hydra
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf
from rdkit.Chem import GetPeriodicTable
from tqdm import tqdm

# Resolvers required by the configs
OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

# QM9 property names in the order stored in .y after dataset reordering:
#   y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
# Original 19 targets: mu alpha homo lumo gap r2 zpve u0 u298 h298 g298 cv
#                      u0_atom u298_atom h298_atom g298_atom A B C
_QM9_PROPERTY_NAMES = [
    "lumo", "gap", "r2", "zpve",
    "u0", "u298", "h298", "g298", "cv",
    "u0_atom", "u298_atom", "h298_atom", "g298_atom",
    "A", "B", "C",
    "mu", "alpha", "homo",
]

_PERIODIC_TABLE = GetPeriodicTable()


def _mw_from_z(z: torch.Tensor) -> float:
    return sum(_PERIODIC_TABLE.GetAtomicWeight(int(zi)) for zi in z.tolist())


def _get_item(dataset, i: int):
    """Return a raw Data object, bypassing FlowMatching transforms."""
    if hasattr(dataset, "get"):
        return dataset.get(i)
    return dataset[i]


def _property_names(n_y: int) -> list[str]:
    if n_y == len(_QM9_PROPERTY_NAMES):
        return _QM9_PROPERTY_NAMES
    return [f"y_{i}" for i in range(n_y)]


def collect(dataset) -> dict[str, np.ndarray]:
    mw_list: list[float] = []
    y_list: list[np.ndarray] = []

    for i in tqdm(range(len(dataset)), desc="Collecting", unit="mol"):
        data = _get_item(dataset, i)
        mw_list.append(_mw_from_z(data.z))

        if hasattr(data, "y") and data.y is not None:
            y_list.append(data.y.view(-1).numpy())

    arrays: dict[str, np.ndarray] = {"mw": np.array(mw_list)}

    if y_list:
        y_arr = np.stack(y_list, axis=0)  # (N, n_props)
        for j, name in enumerate(_property_names(y_arr.shape[1])):
            arrays[name] = y_arr[:, j]

    return arrays


def plot(arrays: dict[str, np.ndarray], out_path: str, n_bins: int = 60) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(arrays.keys())
    n = len(keys)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, key in zip(axes_flat, keys):
        vals = arrays[key]
        ax.hist(vals, bins=n_bins, edgecolor="none", color="steelblue", alpha=0.85)
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        mu = vals.mean()
        ax.axvline(mu, color="red", linestyle="--", linewidth=1, label=f"μ={mu:.3g}")
        ax.legend(fontsize=8)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    # Default output prefix: next to the processed data directory
    out_prefix = cfg.get("out_prefix", None)
    if out_prefix is None:
        processed_root = os.path.dirname(cfg.data.preprocessing.distributions_path)
        out_prefix = os.path.join(processed_root, "property_distributions")

    npz_path = f"{out_prefix}_properties.npz"
    png_path = f"{out_prefix}_distributions.png"

    print("Instantiating train dataset from cfg.data.preprocessing.train_dataset ...")
    dataset = hydra.utils.instantiate(cfg.data.preprocessing.train_dataset)
    print(f"Dataset size: {len(dataset)}")

    arrays = collect(dataset)

    np.savez(npz_path, **arrays)
    print(f"Properties saved to {npz_path}")

    print("\nSummary:")
    for name, arr in arrays.items():
        print(f"  {name:20s}  mean={arr.mean():.4g}  std={arr.std():.4g}"
              f"  min={arr.min():.4g}  max={arr.max():.4g}")

    plot(arrays, png_path)


if __name__ == "__main__":
    main()
