from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA

from .array_utils import normalize_rows
from .fingerprints import FingerprintResult


@dataclass(slots=True)
class PhaseScreeningResult:
    labels: np.ndarray
    embedding: np.ndarray
    pca_components: np.ndarray
    nmf_components: np.ndarray | None
    representative_profiles: np.ndarray
    candidate_scores: dict[str, np.ndarray]
    low_confidence_mask: np.ndarray
    output_dir: Path | None = None


def screen_phases(
    fingerprints: FingerprintResult,
    method: str = "pca_nmf_cluster",
    candidate_phases: list[dict[str, Any]] | None = None,
    *,
    n_components: int = 3,
    n_clusters: int = 3,
    output_dir: str | Path | None = None,
) -> PhaseScreeningResult:
    """Cluster radial fingerprints and optionally score candidate phase profiles."""
    profiles = fingerprints.profiles
    nav_shape = profiles.shape[:2]
    matrix = profiles.reshape(-1, profiles.shape[-1]).astype(np.float32)
    matrix = matrix / np.maximum(matrix.max(axis=1, keepdims=True), 1e-12)

    n_components = max(1, min(int(n_components), matrix.shape[1], matrix.shape[0]))
    pca = PCA(n_components=n_components, random_state=0)
    embedding = pca.fit_transform(matrix)

    nmf_components = None
    if "nmf" in method:
        nmf = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=500)
        nmf.fit(np.clip(matrix, 0, None))
        nmf_components = nmf.components_

    n_clusters = max(1, min(int(n_clusters), matrix.shape[0]))
    labels_flat = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(embedding)
    labels = labels_flat.reshape(nav_shape)

    representative_profiles = np.vstack([
        matrix[labels_flat == label].mean(axis=0) if np.any(labels_flat == label) else np.zeros(matrix.shape[1])
        for label in range(n_clusters)
    ]).astype(np.float32)

    candidate_scores = _score_candidates(matrix, nav_shape, candidate_phases)
    confidence = _cluster_confidence(embedding, labels_flat, n_clusters).reshape(nav_shape)
    low_confidence_mask = confidence < np.percentile(confidence, 15)

    result = PhaseScreeningResult(
        labels=labels.astype(np.int16),
        embedding=embedding.reshape(nav_shape + (embedding.shape[-1],)),
        pca_components=pca.components_.astype(np.float32),
        nmf_components=nmf_components.astype(np.float32) if nmf_components is not None else None,
        representative_profiles=representative_profiles,
        candidate_scores=candidate_scores,
        low_confidence_mask=low_confidence_mask,
        output_dir=Path(output_dir) if output_dir else None,
    )
    if output_dir:
        _save_phase_result(result)
    return result


def _score_candidates(matrix: np.ndarray, nav_shape: tuple[int, int], candidate_phases: list[dict[str, Any]] | None) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    if not candidate_phases:
        return scores

    norm_matrix = normalize_rows(matrix)
    for idx, phase in enumerate(candidate_phases):
        name = str(phase.get("name", f"phase_{idx}"))
        reference = phase.get("reference_profile")
        if reference is None:
            continue
        ref = np.asarray(reference, dtype=np.float32).reshape(1, -1)
        if ref.shape[1] != matrix.shape[1]:
            continue
        score = norm_matrix @ normalize_rows(ref).T
        scores[name] = score.reshape(nav_shape).astype(np.float32)
    return scores


def _cluster_confidence(embedding: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    centers = np.vstack([
        embedding[labels == label].mean(axis=0) if np.any(labels == label) else np.zeros(embedding.shape[1])
        for label in range(n_clusters)
    ])
    distances = np.linalg.norm(embedding - centers[labels], axis=1)
    return 1.0 / (1.0 + distances)


def _save_phase_result(result: PhaseScreeningResult) -> None:
    assert result.output_dir is not None
    result.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(result.output_dir / "phase_labels.npy", result.labels)
    np.save(result.output_dir / "phase_embedding.npy", result.embedding)
    np.save(result.output_dir / "phase_representative_profiles.npy", result.representative_profiles)
    np.save(result.output_dir / "phase_low_confidence_mask.npy", result.low_confidence_mask)
    for name, score in result.candidate_scores.items():
        np.save(result.output_dir / f"candidate_score_{name}.npy", score)
