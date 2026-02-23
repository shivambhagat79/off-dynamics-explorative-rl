"""
Cross-policy t-SNE visualisation for off-dynamics RL experiments.

Usage examples
--------------
# Compare all policies for a given environment:
    python plot_tsne.py halfcheetah-friction-0.1

# With custom t-SNE parameters:
    python plot_tsne.py halfcheetah-morph-foot-hard --max_samples 30000 --perplexity 50

Directory layout expected under ``tsne_data/``:
    tsne_data/<env_name>/<POLICY>/r<seed>/
        metadata.json        – {"state_dim": <int>}
        target_states.bin    – float32 binary of visited next-states

Every policy — including the virtual "UNTUNED" baseline produced by
TUNE_TB — follows this same uniform structure.

The script discovers every policy directory automatically.  When multiple
seeds are available for a policy, one is chosen at random and shown in
the legend so the reader knows which seed produced the plot.

Output is saved to:
    tsne_plots/<env_name>/policy_comparison_tsne.png
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Colour palette – visually distinct, colour-blind friendly
_COLOURS = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


def _load_bin(path, state_dim):
    """Load a flat float32 binary file into shape ``(N, state_dim)``."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.empty((0, state_dim), dtype=np.float32)
    return raw.reshape(-1, state_dim)


def _subsample(arr, max_n, rng):
    if arr.shape[0] <= max_n:
        return arr
    idx = rng.choice(arr.shape[0], max_n, replace=False)
    return arr[idx]


def _pick_seed(seed_dirs):
    """Pick one seed directory at random and return (seed_dir_path, seed_int)."""
    chosen = random.choice(seed_dirs)
    seed_name = os.path.basename(chosen)
    # seed directories are named "r0", "r1", etc.
    seed_int = int(seed_name[1:])
    return chosen, seed_int


def discover_data(tsne_root, env_name):
    """
    Walk ``tsne_root/<env_name>/`` and return a mapping from a legend label
    to a ``(N, state_dim)`` numpy array of states.

    Every policy directory is expected to contain seed sub-directories
    (e.g. ``r0``, ``r1``), each holding ``metadata.json`` and
    ``target_states.bin``.

    When a policy has multiple seeds, one is chosen at random.
    Legend labels follow the format ``{POLICY}_{env_name}_r{seed}``.
    """
    base = os.path.join(tsne_root, env_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"No t-SNE data directory found at '{base}'. "
            "Run training with --tsne first."
        )

    datasets = {}

    for policy_name in sorted(os.listdir(base)):
        policy_dir = os.path.join(base, policy_name)
        if not os.path.isdir(policy_dir):
            continue

        # Collect all valid seed directories for this policy
        seed_dirs = []
        for entry in sorted(os.listdir(policy_dir)):
            if not entry.startswith("r"):
                continue
            candidate = os.path.join(policy_dir, entry)
            if os.path.isdir(candidate) and os.path.isfile(
                os.path.join(candidate, "metadata.json")
            ):
                seed_dirs.append(candidate)

        if not seed_dirs:
            print(f"  [skip] {policy_dir}: no valid seed directories found")
            continue

        # Pick one seed at random
        seed_dir, seed_int = _pick_seed(seed_dirs)

        meta_path = os.path.join(seed_dir, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        state_dim = int(meta["state_dim"])

        target_path = os.path.join(seed_dir, "target_states.bin")
        if not os.path.isfile(target_path):
            print(f"  [skip] {seed_dir}: target_states.bin not found")
            continue

        data = _load_bin(target_path, state_dim)
        if data.shape[0] == 0:
            print(f"  [warn] {policy_name} (seed={seed_int}): 0 states – skipping")
            continue

        datasets[f"{policy_name}_{env_name}_r{seed_int}"] = data

    return datasets


def plot(
    datasets, env_name, output_dir, max_samples=50000, perplexity=30, random_state=42
):
    """
    Run joint t-SNE on *all* policy datasets and save a comparative scatter
    plot.  Returns the path to the saved image.
    """
    rng = np.random.default_rng(random_state)

    # Subsample each dataset independently
    labels_order = sorted(datasets.keys())
    subsampled = {k: _subsample(datasets[k], max_samples, rng) for k in labels_order}

    # Build combined matrix and record boundaries
    parts = []
    boundaries = []  # (label, start, end)
    offset = 0
    for label in labels_order:
        arr = subsampled[label]
        parts.append(arr)
        boundaries.append((label, offset, offset + arr.shape[0]))
        offset += arr.shape[0]

    combined = np.concatenate(parts, axis=0)
    print(
        f"Running t-SNE on {combined.shape[0]} states ({len(labels_order)} variants) …"
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedded = tsne.fit_transform(combined)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, (label, start, end) in enumerate(boundaries):
        colour = _COLOURS[idx % len(_COLOURS)]
        n_pts = end - start
        ax.scatter(
            embedded[start:end, 0],
            embedded[start:end, 1],
            s=5,
            alpha=0.4,
            label=f"{label} ({n_pts} states)",
            color=colour,
        )

    ax.set_title(f"t-SNE State-Space Coverage — {env_name}")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(markerscale=4, loc="best", fontsize="small")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "policy_comparison_tsne.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate cross-policy t-SNE comparison plots from saved state data. "
            "Example: python plot_tsne.py halfcheetah-friction-0.1"
        ),
    )
    parser.add_argument(
        "env",
        type=str,
        help=(
            "Full environment identifier (env + shift level), e.g. "
            "'halfcheetah-friction-0.1' or 'halfcheetah-morph-foot-hard'. "
            "Dashes are converted to underscores to locate the data directory."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Max states per policy variant to feed into t-SNE (default: 50000).",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter (default: 30).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for t-SNE reproducibility (default: 42).",
    )

    args = parser.parse_args()

    # Normalise env name: replace dashes with underscores to match directory names
    env_name = args.env.replace("-", "_")

    tsne_data_dir = "./tsne_data"
    output_dir = os.path.join("./tsne_plots", env_name)

    print(f"Looking for t-SNE data in: {tsne_data_dir}/{env_name}/")
    datasets = discover_data(tsne_data_dir, env_name)

    if not datasets:
        print("No state data found. Nothing to plot.")
        return

    for label, arr in sorted(datasets.items()):
        print(f"  {label}: {arr.shape[0]} states (dim={arr.shape[1]})")

    save_path = plot(
        datasets,
        env_name,
        output_dir,
        max_samples=args.max_samples,
        perplexity=args.perplexity,
        random_state=args.random_state,
    )
    print(f"t-SNE plot saved to {save_path}")


if __name__ == "__main__":
    main()
