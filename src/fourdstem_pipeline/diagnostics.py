from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .array_utils import as_numpy_block, iter_navigation_slices, normalize_rows
from .dataset import DatasetHandle
from .export import save_bar_png, save_label_png, save_lines_png, save_png
from .fingerprints import FingerprintResult
from .orientation import OrientationResult
from .phase import PhaseScreeningResult
from .virtual import VirtualImageResult


def run_stage1_diagnostics(
    dataset: DatasetHandle,
    fingerprints: FingerprintResult,
    phase: PhaseScreeningResult,
    virtual: VirtualImageResult,
    orientation: OrientationResult,
    *,
    output_dir: str | Path,
    png_dir: str | Path,
    block_shape: tuple[int, int],
    confidence_threshold: float,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    png_dir = Path(png_dir)
    cluster_dir = output_dir / "05_cluster_diagnostics"
    roi_dir = output_dir / "06_roi_candidates"
    class_dir = output_dir / "03_diffraction_classes"
    orientation_dir = output_dir / "04_orientation_preview"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    labels = phase.labels
    cluster_ids = sorted(int(v) for v in np.unique(labels))
    cluster_mean_dps = _cluster_mean_diffraction(dataset, labels, cluster_ids, block_shape)
    np.save(cluster_dir / "cluster_mean_dps.npy", cluster_mean_dps)
    for idx, cluster_id in enumerate(cluster_ids):
        save_png(png_dir / f"cluster_mean_dp_{cluster_id}.png", cluster_mean_dps[idx])
        save_png(cluster_dir / f"cluster_mean_dp_{cluster_id}.png", cluster_mean_dps[idx])
        save_png(png_dir / f"cluster_mean_dp_log_{cluster_id}.png", np.log1p(cluster_mean_dps[idx]))
        save_png(cluster_dir / f"cluster_mean_dp_log_{cluster_id}.png", np.log1p(cluster_mean_dps[idx]))

    radial_mean, radial_std = _cluster_radial_profiles(fingerprints.profiles, labels, cluster_ids)
    np.save(cluster_dir / "cluster_mean_radial_profiles.npy", radial_mean)
    np.save(cluster_dir / "cluster_radial_profile_std.npy", radial_std)
    save_lines_png(png_dir / "cluster_mean_radial_profiles.png", fingerprints.radii, radial_mean)
    save_lines_png(cluster_dir / "cluster_mean_radial_profiles.png", fingerprints.radii, radial_mean)
    save_lines_png(png_dir / "cluster_radial_profile_mean_std.png", fingerprints.radii, _interleave_mean_std(radial_mean, radial_std))
    save_lines_png(cluster_dir / "cluster_radial_profile_mean_std.png", fingerprints.radii, _interleave_mean_std(radial_mean, radial_std))

    stats_rows = _cluster_virtual_statistics(labels, virtual, cluster_ids)
    _write_cluster_summary(cluster_dir, stats_rows)
    save_bar_png(png_dir / "cluster_virtual_image_statistics.png", _stats_bar_values(stats_rows))
    save_bar_png(cluster_dir / "cluster_virtual_image_statistics.png", _stats_bar_values(stats_rows))

    norm_outputs = _normalization_comparison(fingerprints.profiles, len(cluster_ids), class_dir, png_dir)
    k_sweep_outputs = _k_sweep(fingerprints.profiles, [2, 3, 4, 5, 6, 8], class_dir, png_dir)
    beam_outputs = _beam_diagnostics(virtual, png_dir, output_dir / "00_preprocess")
    component_outputs = _connected_component_diagnostics(labels, virtual.images, cluster_ids, cluster_dir, png_dir)
    orientation_outputs = _orientation_reliability(orientation, confidence_threshold, png_dir, orientation_dir)
    roi_outputs = _roi_candidates(labels, virtual.images, orientation, cluster_ids, roi_dir, png_dir)

    return {
        "cluster_diagnostics": str(cluster_dir),
        "roi_candidates": str(roi_dir),
        "cluster_summary_csv": str(cluster_dir / "cluster_summary.csv"),
        "cluster_summary_md": str(cluster_dir / "cluster_summary.md"),
        "normalization_comparison": norm_outputs,
        "k_sweep": k_sweep_outputs,
        "beam": beam_outputs,
        "connected_components": component_outputs,
        "orientation_reliability": orientation_outputs,
        "roi_outputs": roi_outputs,
    }


def _cluster_mean_diffraction(dataset: DatasetHandle, labels: np.ndarray, cluster_ids: list[int], block_shape: tuple[int, int]) -> np.ndarray:
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


def _cluster_radial_profiles(profiles: np.ndarray, labels: np.ndarray, cluster_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    matrix = profiles.reshape(-1, profiles.shape[-1]).astype(np.float32)
    flat_labels = labels.reshape(-1)
    means = []
    stds = []
    for cluster_id in cluster_ids:
        selected = matrix[flat_labels == cluster_id]
        means.append(selected.mean(axis=0) if selected.size else np.zeros(matrix.shape[1], dtype=np.float32))
        stds.append(selected.std(axis=0) if selected.size else np.zeros(matrix.shape[1], dtype=np.float32))
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def _interleave_mean_std(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    lines = []
    for idx in range(mean.shape[0]):
        lines.extend([mean[idx], mean[idx] + std[idx], np.maximum(mean[idx] - std[idx], 0)])
    return np.asarray(lines, dtype=np.float32)


def _cluster_virtual_statistics(labels: np.ndarray, virtual: VirtualImageResult, cluster_ids: list[int]) -> list[dict[str, float | int]]:
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


def _write_cluster_summary(output_dir: Path, rows: list[dict[str, float | int]]) -> None:
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


def _stats_bar_values(rows: list[dict[str, float | int]]) -> np.ndarray:
    keys = ["mean_BF", "mean_ADF", "mean_HAADF", "mean_ring_1", "mean_ring_2", "mean_ring_3"]
    return np.asarray([[float(row.get(key, 0.0)) for key in keys] for row in rows], dtype=np.float32)


def _normalization_comparison(profiles: np.ndarray, n_clusters: int, class_dir: Path, png_dir: Path) -> dict[str, str]:
    matrix = profiles.reshape(-1, profiles.shape[-1]).astype(np.float32)
    variants = {
        "raw": matrix,
        "l1norm": matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12),
        "log_zscore": _zscore(np.log1p(matrix)),
    }
    outputs = {}
    for name, values in variants.items():
        labels = _cluster_matrix(values, profiles.shape[:2], n_clusters)
        np.save(class_dir / f"labels_{name}.npy", labels)
        save_label_png(png_dir / f"diffraction_class_labels_{name}.png", labels)
        outputs[name] = str(class_dir / f"labels_{name}.npy")
    return outputs


def _k_sweep(profiles: np.ndarray, ks: list[int], class_dir: Path, png_dir: Path) -> dict[str, str]:
    matrix = profiles.reshape(-1, profiles.shape[-1]).astype(np.float32)
    matrix = matrix / np.maximum(matrix.max(axis=1, keepdims=True), 1e-12)
    pca_dim = min(6, matrix.shape[1], matrix.shape[0])
    embedding = PCA(n_components=pca_dim, random_state=0).fit_transform(matrix)
    sample = _metric_sample(embedding)
    rows = []
    for k in ks:
        labels_flat = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(embedding)
        labels = labels_flat.reshape(profiles.shape[:2])
        np.save(class_dir / f"labels_k{k}.npy", labels)
        save_label_png(png_dir / f"diffraction_class_labels_k{k}.png", labels)
        metric_labels = labels_flat[sample]
        metric_embedding = embedding[sample]
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(metric_embedding, metric_labels)) if len(np.unique(metric_labels)) > 1 else float("nan"),
                "calinski_harabasz": float(calinski_harabasz_score(metric_embedding, metric_labels)) if len(np.unique(metric_labels)) > 1 else float("nan"),
                "davies_bouldin": float(davies_bouldin_score(metric_embedding, metric_labels)) if len(np.unique(metric_labels)) > 1 else float("nan"),
                "cluster_size_fractions": ";".join(f"{v:.4f}" for v in np.bincount(labels_flat, minlength=k) / labels_flat.size),
                "spatial_fragmentation": float(_fragmentation(labels)),
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


def _cluster_matrix(matrix: np.ndarray, nav_shape: tuple[int, int], n_clusters: int) -> np.ndarray:
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
    total_components = 0
    for label in np.unique(labels):
        total_components += len(_connected_components(labels == label))
    return total_components / max(labels.size, 1)


def _beam_diagnostics(virtual: VirtualImageResult, png_dir: Path, preprocess_dir: Path) -> dict[str, str]:
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    mean_dp = virtual.mean_diffraction
    max_dp = virtual.max_diffraction
    yy, xx = np.indices(mean_dp.shape)
    total = max(float(mean_dp.sum()), 1e-12)
    cy = float((mean_dp * yy).sum() / total)
    cx = float((mean_dp * xx).sum() / total)
    radial_center = ((mean_dp.shape[0] - 1) / 2, (mean_dp.shape[1] - 1) / 2)
    offset = float(np.hypot(cy - radial_center[0], cx - radial_center[1]))
    text = f"estimated_center_yx: [{cy:.3f}, {cx:.3f}]\nradial_center_yx: [{radial_center[0]:.3f}, {radial_center[1]:.3f}]\noffset_pixels: {offset:.3f}\n"
    if offset > 3:
        text += "WARNING: Estimated beam center is offset from radial integration center. Radial fingerprints and diffraction-class labels may be biased.\n"
    (preprocess_dir / "beam_center_estimate.txt").write_text(text, encoding="utf-8")
    save_png(png_dir / "mean_dp_log.png", np.log1p(mean_dp))
    save_png(png_dir / "max_dp_log.png", np.log1p(max_dp))
    save_png(png_dir / "mean_dp_with_center_marker.png", _center_overlay(mean_dp, cy, cx))
    save_png(png_dir / "mean_dp_with_radial_bins.png", _radial_overlay(mean_dp, radial_center))
    save_png(png_dir / "radial_mask_overlay.png", _radial_overlay(mean_dp, radial_center))
    central_mask = ((yy - radial_center[0]) ** 2 + (xx - radial_center[1]) ** 2) <= 7**2
    save_png(png_dir / "central_disk_mask.png", central_mask)
    saturation = (max_dp >= np.percentile(max_dp, 99.9)).astype(np.float32)
    save_png(png_dir / "saturation_fraction_map.png", saturation)
    return {"beam_center": str(preprocess_dir / "beam_center_estimate.txt")}


def _connected_component_diagnostics(labels: np.ndarray, images: dict[str, np.ndarray], cluster_ids: list[int], output_dir: Path, png_dir: Path) -> dict[str, str]:
    rows = []
    cleaned = labels.copy()
    for cluster_id in cluster_ids:
        comps = _connected_components(labels == cluster_id)
        sizes = np.asarray([len(comp) for comp in comps], dtype=np.int64)
        small = int(np.sum(sizes < 16)) if sizes.size else 0
        largest = int(sizes.max()) if sizes.size else 0
        rows.append(
            {
                "cluster": cluster_id,
                "component_count": len(comps),
                "largest_component_size": largest,
                "largest_component_fraction_within_class": largest / max(int(np.sum(labels == cluster_id)), 1),
                "median_component_size": float(np.median(sizes)) if sizes.size else 0.0,
                "small_component_count": small,
            }
        )
        for comp, size in zip(comps, sizes):
            if size < 16:
                for y, x in comp:
                    cleaned[y, x] = _majority_neighbor(labels, y, x)
    with (output_dir / "cluster_connected_components.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_label_png(png_dir / "cluster_cleaned_labels.png", cleaned)
    save_bar_png(png_dir / "cluster_area_histogram.png", np.asarray([[row["largest_component_size"], row["component_count"]] for row in rows], dtype=np.float32))
    base = images.get("adf") if "adf" in images else next(iter(images.values()))
    save_png(png_dir / "cluster_boundary_overlay_on_adf.png", _boundary_overlay(base, labels))
    return {"connected_components_csv": str(output_dir / "cluster_connected_components.csv")}


def _orientation_reliability(orientation: OrientationResult, threshold: float, png_dir: Path, output_dir: Path) -> dict[str, str]:
    unreliable = orientation.score < threshold
    masked_score = orientation.score.copy()
    masked_index = orientation.orientation_index.copy()
    masked_score[unreliable] = 0
    masked_index[unreliable] = -1
    np.save(output_dir / "orientation_score_masked.npy", masked_score)
    np.save(output_dir / "orientation_index_masked.npy", masked_index)
    save_png(png_dir / "orientation_score_masked.png", masked_score)
    save_label_png(png_dir / "orientation_index_masked.png", np.maximum(masked_index, 0))
    hist, _ = np.histogram(orientation.score, bins=32)
    save_bar_png(png_dir / "orientation_score_histogram.png", hist)
    return {"masked_score": str(output_dir / "orientation_score_masked.npy")}


def _roi_candidates(labels: np.ndarray, images: dict[str, np.ndarray], orientation: OrientationResult, cluster_ids: list[int], output_dir: Path, png_dir: Path) -> dict[str, str]:
    rois: list[dict[str, Any]] = []
    for cluster_id in cluster_ids:
        comps = _connected_components(labels == cluster_id)
        if not comps:
            continue
        largest = max(comps, key=len)
        rois.append(_roi_from_component(f"cluster{cluster_id}_core_01", largest, labels.shape, cluster=cluster_id, reason="largest connected component core", size=64))
        if np.sum(labels == cluster_id) / labels.size < 0.15:
            rois.append(_roi_from_component(f"cluster{cluster_id}_minority_01", largest, labels.shape, cluster=cluster_id, reason="minority diffraction class", size=32))
    boundary = np.argwhere(_boundary_mask(labels))
    if boundary.size:
        center = boundary[len(boundary) // 2]
        rois.append(_roi_from_center("boundary_classes_01", int(center[1]), int(center[0]), labels.shape, 64, "class boundary", None))
    rois.extend(_orientation_rois(orientation, labels.shape))
    rois.extend(_intensity_rois(images, labels.shape))

    yaml_path = output_dir / "roi_candidates.yaml"
    csv_path = output_dir / "roi_candidates.csv"
    yaml_path.write_text(yaml.safe_dump({"rois": rois}, sort_keys=False), encoding="utf-8")
    keys = sorted({key for roi in rois for key in roi})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: roi.get(key, "") for key in keys} for roi in rois])
    base = images.get("adf") if "adf" in images else next(iter(images.values()))
    save_png(png_dir / "roi_candidates_overlay.png", _roi_overlay(base, rois))
    save_png(output_dir / "roi_candidates_overlay.png", _roi_overlay(base, rois))
    return {"yaml": str(yaml_path), "csv": str(csv_path)}


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        comp: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            comp.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        components.append(comp)
    return components


def _roi_from_component(name: str, component: list[tuple[int, int]], shape: tuple[int, int], *, cluster: int, reason: str, size: int) -> dict[str, Any]:
    coords = np.asarray(component)
    y = int(np.median(coords[:, 0]))
    x = int(np.median(coords[:, 1]))
    return _roi_from_center(name, x, y, shape, size, reason, cluster)


def _roi_from_center(name: str, x: int, y: int, shape: tuple[int, int], size: int, reason: str, cluster: int | None) -> dict[str, Any]:
    half = size // 2
    x0 = max(0, min(x - half, shape[1] - size))
    y0 = max(0, min(y - half, shape[0] - size))
    x1 = min(shape[1], x0 + size)
    y1 = min(shape[0], y0 + size)
    roi: dict[str, Any] = {
        "name": name,
        "center": [int((x0 + x1) / 2), int((y0 + y1) / 2)],
        "bbox": [int(x0), int(x1), int(y0), int(y1)],
        "size": [int(x1 - x0), int(y1 - y0)],
        "reason": reason,
        "recommended_stage2": "py4dstem_bragg_indexing",
    }
    if cluster is not None:
        roi["cluster"] = int(cluster)
    return roi


def _orientation_rois(orientation: OrientationResult, label_shape: tuple[int, int]) -> list[dict[str, Any]]:
    scale_y = label_shape[0] / orientation.score.shape[0]
    scale_x = label_shape[1] / orientation.score.shape[1]
    high = np.unravel_index(int(np.argmax(orientation.score)), orientation.score.shape)
    low = np.unravel_index(int(np.argmin(orientation.score)), orientation.score.shape)
    return [
        _roi_from_center("high_orientation_score_01", int((high[1] + 0.5) * scale_x), int((high[0] + 0.5) * scale_y), label_shape, 64, "high orientation score", None),
        _roi_from_center("low_orientation_score_01", int((low[1] + 0.5) * scale_x), int((low[0] + 0.5) * scale_y), label_shape, 64, "low orientation score", None),
    ]


def _intensity_rois(images: dict[str, np.ndarray], shape: tuple[int, int]) -> list[dict[str, Any]]:
    rois = []
    for name in ("adf", "haadf"):
        if name in images:
            y, x = np.unravel_index(int(np.argmax(images[name])), images[name].shape)
            rois.append(_roi_from_center(f"{name}_high_intensity_01", int(x), int(y), shape, 64, f"{name.upper()} intensity anomaly", None))
    return rois


def _majority_neighbor(labels: np.ndarray, y: int, x: int) -> int:
    y0, y1 = max(0, y - 1), min(labels.shape[0], y + 2)
    x0, x1 = max(0, x - 1), min(labels.shape[1], x + 2)
    values, counts = np.unique(labels[y0:y1, x0:x1], return_counts=True)
    return int(values[np.argmax(counts)])


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=bool)
    mask[:-1, :] |= labels[:-1, :] != labels[1:, :]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return mask


def _scale_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(arr[np.isfinite(arr)], (1, 99))
    return np.clip((arr - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)


def _boundary_overlay(base: np.ndarray, labels: np.ndarray) -> np.ndarray:
    gray = _scale_gray(base)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[_boundary_mask(labels)] = [235, 40, 40]
    return rgb


def _roi_overlay(base: np.ndarray, rois: list[dict[str, Any]]) -> np.ndarray:
    gray = _scale_gray(base)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    for roi in rois:
        x0, x1, y0, y1 = [int(v) for v in roi["bbox"]]
        rgb[y0:y1, x0] = [255, 215, 0]
        rgb[y0:y1, max(x1 - 1, x0)] = [255, 215, 0]
        rgb[y0, x0:x1] = [255, 215, 0]
        rgb[max(y1 - 1, y0), x0:x1] = [255, 215, 0]
    return rgb


def _center_overlay(image: np.ndarray, cy: float, cx: float) -> np.ndarray:
    gray = _scale_gray(image)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    y = int(round(cy))
    x = int(round(cx))
    rgb[max(0, y - 5):min(rgb.shape[0], y + 6), x] = [255, 40, 40]
    rgb[y, max(0, x - 5):min(rgb.shape[1], x + 6)] = [255, 40, 40]
    return rgb


def _radial_overlay(image: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    gray = _scale_gray(image)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    yy, xx = np.indices(image.shape)
    rr = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    for radius in np.linspace(8, min(image.shape) / 2 - 2, 5):
        rgb[np.abs(rr - radius) < 0.5] = [40, 200, 80]
    return rgb
