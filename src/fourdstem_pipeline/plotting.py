from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_image_map(image: np.ndarray, path: str | Path, *, title: str, cmap: str = "viridis") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_profile_plot(profiles: np.ndarray, path: str | Path, *, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for profile in np.asarray(profiles):
        ax.plot(profile)
    ax.set_title(title)
    ax.set_xlabel("Radial bin")
    ax.set_ylabel("Mean intensity")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
