"""Cluster-level diagnostics: mean diffraction patterns, radial profiles,
virtual-image statistics, normalisation comparison, and K-sweep metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .array_utils import as_numpy_block, iter_navigation_slices
from .dataset import DatasetHandle
from .export import save_bar_png, save_label_png, save_lines_png, save_png
from .fingerprints import FingerprintResult
from .virtual import VirtualImageResult


# ---------------------------------------------------------------------------
# Cluster mean diffraction patterns
# ---------------------------------------------------------------------------


def cluster_mean_diffraction(
    dataset: DatasetHandle,
    labels: np.ndarray,
    cluster_ids: list[int],
    block_shape: tuple[int, int],
) -> np.ndarray:
    sums = np.zeros((len(cluster_ids),) + dataset.signal_shape, dtype=np.float64)
    counts = np.zeros(len(cluster_ids), dtype=np.int64)
    id_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}
    for ys, xs in iter_navigation_slices(dataset.navigation_shape, block_shape):
        block = as_numpy_block(dataset.data[ys, xs, :, :]).astype(np.float32, copy=False)
        block_labels = labels[ys, xs]
        for cluster_id in cluster_ids:
            mask = block_labels == cluster_id
            if np.any(mask):
                idx = id_to_idx[cluster_id]
                sums[idx] += block[mask].sum(axis=0)
                counts[idx] += int(mask.sum())
    return (sums / np.maximum(counts[:, None, None], 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Cluster radial profiles
# ---------------------------------------------------------------------------


def cluster_radial_profiles(
    profiles: np.ndarray,
    labels: np.ndarray,
    cluster_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = profiles.reshape(-1, profiles.shape[-1]).astype(np.float32)
    flat_labels = labels.reshape(-1)
    means = []
    stds = []
    for cluster_id in cluster_ids:
        selected = matrix[flat_labels == cluster_id]
        means.append(selected.mean(axis=0) if selected.size else np.zeros(matrix.shape[1], dtype=np.float32))
        stds.append(selected.std(axis=0) if selected.size else np.zeros(matrix.shape[1], dtype=np.float32))
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def interleave_mean_std(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    lines = []
    for idx in range(mean.shape[0]):
        lines.extend([mean[idx], mean[idx] + std[idx], np.maximum(mean[idx] - std[idx], 0)])
    return np.asarray(lines, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cluster virtual-image statistics
# ---------------------------------------------------------------------------


def cluster_virtual_statistics(
    labels: np.ndarray,
    virtual: VirtualImageResult,
    cluster_ids: list[int],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    total = max(labels.size, 1)
    image_names = ["bf", "adf", "haadf", "ring_1", "ring_2", "ring_3"]
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        row: dict[str, float | int] = {
            "cluster": cluster_id,
            "pixel_count": int(mask.sum()),
            "fraction": float(mask.sum() / total),
        }
        for name in image_names:
            image = virtual.images.get(name)
            key = name.upper() if name in {"bf", "adf", "haadf"} else name
            if image is not None:
                row[f"mean_{key}"] = float(np.mean(image[mask]))
                row[f"std_{key}"] = float(np.std(image[mask]))
        row["mean_COM_x"] = float(np.mean(virtual.com_x[mask]))
        row["mean_COM_y"] = float(np.mean(virtual.com_y[mask]))
        row["std_COM_x"] = float(np.std(virtual.com_x[mask]))
        row["std_COM_y"] = float(np.std(virtual.com_y[mask]))
        row["ring_2/ring_1"] = _safe_ratio(row.get("mean_ring_2"), row.get("mean_ring_1"))
        row["ring_3/ring_1"] = _safe_ratio(row.get("mean_ring_3"), row.get("mean_ring_1"))
        row["ring_3/ring_2"] = _safe_ratio(row.get("mean_ring_3"), row.get("mean_ring_2"))
        row["ADF/BF"] = _safe_ratio(row.get("mean_ADF"), row.get("mean_BF"))
        row["HAADF/ADF"] = _safe_ratio(row.get("mean_HAADF"), row.get("mean_ADF"))
        rows.append(row)
    return rows


def _safe_ratio(num: Any, den: Any) -> float:
    try:
        return float(num) / max(float(den), 1e-12)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Summary file output (CSV + Markdown)
# ---------------------------------------------------------------------------


def write_cluster_summary(output_dir: Path, rows: list[dict[str, float | int]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with (output_dir / "cluster_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(key, "")) for key in keys) + " |")
    (output_dir / "cluster_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def stats_bar_values(rows: list[dict[str, float | int]]) -> np.ndarray:
    keys = ["mean_BF", "mean_ADF", "mean_HAADF", "mean_ring_1", "mean_ring_2", "mean_ring_3"]
    return np.asarray([[float(row.get(key, 0.0)) for key in keys] for row in rows], dtype=np.float32)


# ---------------------------------------------------------------------------
# Normalisation comparison (raw / L1-norm / log-zscore)
# ---------------------------------------------------------------------------


def normalisation_comparison(
    profiles: np.ndarray,
    n_clusters: int,
    class_dir: Path,
    png_dir: Path,
) -> dict[str, str]:
    matrix = profiles.reshape(-1, profiles.shape[-1]).astype(np.float32)
    variants = {
        "raw": matrix,
        "l1norm": matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12),
        "log_zscore": _zscore(np.log1p(matrix)),
    }
    outputs = {}
    for name, values in variants.items():
        labels_result = cluster_matrix(values, profiles.shape[:2], n_clusters)
        np.save(class_dir / f"labels_{name}.npy", labels_result)
        save_label_png(png_dir / f"diffraction_class_labels_{name}.png", labels_result)
        outputs[name] = str(class_dir / f"labels_{name}.npy")
    return outputs


# ---------------------------------------------------------------------------
# K-sweep metrics
# ---------------------------------------------------------------------------


def k_sweep(
    profiles: np.ndarray,
    ks: list[int],
    class_dir: Path,
    png_dir: Path,
) -> dict[str, str]:
    matrix = profiles.reshape(-1, profiles.shape[-1]).astype(np.float32)
    matrix = matrix / np.maximum(matrix.max(axis=1, keepdims=True), 1e-12)
    pca_dim = min(6, matrix.shape[1], matrix.shape[0])
    embedding = PCA(n_components=pca_dim, random_state=0).fit_transform(matrix)
    sample = _metric_sample(embedding)
    rows = []
    for k in ks:
        labels_flat = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(embedding)
        labels_result = labels_flat.reshape(profiles.shape[:2])
        np.save(class_dir / f"labels_k{k}.npy", labels_result)
        save_label_png(png_dir / f"diffraction_class_labels_k{k}.png", labels_result)
        metric_labels = labels_flat[sample]
        metric_embedding = embedding[sample]
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(metric_embedding, metric_labels)) if len(np.unique(metric_labels)) > 1 else float("nan"),
                "calinski_harabasz": float(calinski_harabasz_score(metric_embedding, metric_labels)) if len(np.unique(metric_labels)) > 1 else float("nan"),
                "davies_bouldin": float(davies_bouldin_score(metric_embedding, metric_labels)) if len(np.unique(metric_labels)) > 1 else float("nan"),
                "cluster_size_fractions": ";".join(f"{v:.4f}" for v in np.bincount(labels_flat, minlength=k) / labels_flat.size),
                "spatial_fragmentation": float(_fragmentation(labels_result)),
            }
        )
    with (class_dir / "k_sweep_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_lines_png(
        png_dir / "k_sweep_metrics.png",
        np.asarray([row["k"] for row in rows], dtype=np.float32),
        np.asarray([[row["silhouette"] for row in rows], [row["spatial_fragmentation"] for row in rows]], dtype=np.float32),
    )
    return {"metrics_csv": str(class_dir / "k_sweep_metrics.csv"), "metrics_png": str(png_dir / "k_sweep_metrics.png")}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def cluster_matrix(matrix: np.ndarray, nav_shape: tuple[int, int], n_clusters: int) -> np.ndarray:
    pca_dim = min(6, matrix.shape[1], matrix.shape[0])
    embedding = PCA(n_components=pca_dim, random_state=0).fit_transform(matrix)
    return KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(embedding).reshape(nav_shape).astype(np.int16)


def _zscore(matrix: np.ndarray) -> np.ndarray:
    return (matrix - matrix.mean(axis=1, keepdims=True)) / np.maximum(matrix.std(axis=1, keepdims=True), 1e-12)


def _metric_sample(matrix: np.ndarray, max_samples: int = 5000) -> np.ndarray:
    if matrix.shape[0] <= max_samples:
        return np.arange(matrix.shape[0])
    return np.linspace(0, matrix.shape[0] - 1, max_samples).astype(int)


def _fragmentation(labels: np.ndarray) -> float:
    from .diagnostics_spatial import connected_components as cc

    total_components = 0
    for label in np.unique(labels):
        total_components += len(cc(labels == label))
    return total_components / max(labels.size, 1)
