import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne(
    src_states,
    tar_states,
    output_dir,
    max_samples=50000,
    perplexity=30,
    random_state=42,
):
    """
    Generate a t-SNE plot comparing the state-space coverage of the source
    and target environments using online-collected state arrays.

    Args:
        src_states: np.ndarray of shape (N, state_dim) — all states visited
                    in the source environment during training.
        tar_states: np.ndarray of shape (M, state_dim) — all states visited
                    in the target environment during training.
        output_dir: Directory path where the plot will be saved.
        max_samples: Maximum number of states to sample from each array
                     (subsampling avoids excessive t-SNE compute time).
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed for t-SNE reproducibility.
    """
    # Subsample if collected states are large
    if src_states.shape[0] > max_samples:
        idx = np.random.choice(src_states.shape[0], max_samples, replace=False)
        src_states = src_states[idx]
    if tar_states.shape[0] > max_samples:
        idx = np.random.choice(tar_states.shape[0], max_samples, replace=False)
        tar_states = tar_states[idx]

    n_src = src_states.shape[0]
    n_tar = tar_states.shape[0]

    # Concatenate for a joint t-SNE embedding
    combined = np.concatenate([src_states, tar_states], axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedded = tsne.fit_transform(combined)

    src_embedded = embedded[:n_src]
    tar_embedded = embedded[n_src:]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        src_embedded[:, 0],
        src_embedded[:, 1],
        s=5,
        alpha=0.4,
        label=f"Source ({n_src} states)",
        color="#1f77b4",
    )
    ax.scatter(
        tar_embedded[:, 0],
        tar_embedded[:, 1],
        s=5,
        alpha=0.4,
        label=f"Target ({n_tar} states)",
        color="#ff7f0e",
    )

    ax.set_title("t-SNE of State-Space Coverage (Source vs Target)")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(markerscale=4)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "state_tsne.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"t-SNE plot saved to {save_path}")
