"""Minimal TEMM (Trajectory-based Emergent Markov Model) for role alignment.

Implements the core Org-Fit pipeline from Soule et al. 2025 (MOISE+MARL, AAMAS):
agglomerative hierarchical clustering of (obs, action) trajectories +
Hungarian assignment to predefined role labels.

This is a self-contained ~150 LOC reimplementation tailored to SFAC-FACMAC
controller trajectories (obs dim = 5, action dim = 3 per instance).

Exports:
    cluster_trajectories(X, n_clusters)           -> cluster labels (N,)
    hungarian_assignment(true, pred, K)           -> remapped pred (N,)
    micro_f1(true, pred_remapped)                 -> float
    structural_fit(X, labels)                     -> SOF in [0, 1]
    functional_fit(obs_only, labels)              -> FOF in [0, 1]
    org_fit(X, obs_only, true_labels, K)          -> dict with f1/sof/fof/of

Reference: Soule et al. 2025, MOISE+MARL paper, AAMAS proceedings; TEMM module
in marllib_moise_marl/mma/mma_wrapper/temm/.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances


def flatten_trajectory(traj_obs: np.ndarray, traj_act: np.ndarray) -> np.ndarray:
    """Flatten a (T, obs_dim) + (T, act_dim) trajectory into one 1-D vector.

    Used by the agglomerative clustering distance metric.
    """
    return np.concatenate([traj_obs.flatten(), traj_act.flatten()])


def cluster_trajectories(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Agglomerative hierarchical clustering with Ward linkage.

    Args:
        X: (N, D) feature matrix, one row per trajectory.
        n_clusters: number of clusters to extract (= number of roles).

    Returns:
        labels: (N,) integer cluster labels in [1, n_clusters].
    """
    if len(X) <= n_clusters:
        # Degenerate: each trajectory becomes its own cluster
        return np.arange(1, len(X) + 1)
    # Ward linkage minimizes within-cluster variance (used by Soule2025)
    Z = linkage(X, method="ward")
    return fcluster(Z, t=n_clusters, criterion="maxclust")


def hungarian_assignment(true: np.ndarray, pred: np.ndarray,
                         n_classes: int) -> np.ndarray:
    """Optimally remap cluster IDs to ground-truth label IDs via Hungarian.

    Builds a confusion matrix cluster_id x true_label, then finds the
    assignment that maximizes diagonal sum.

    Args:
        true: (N,) ground-truth labels in [0, n_classes).
        pred: (N,) predicted cluster labels (any integer IDs).
        n_classes: number of distinct labels.

    Returns:
        pred_remapped: (N,) predictions remapped to ground-truth label space.
    """
    pred_ids = np.unique(pred)
    cost = np.zeros((len(pred_ids), n_classes))
    for i, pid in enumerate(pred_ids):
        mask = pred == pid
        for c in range(n_classes):
            # Negative because linear_sum_assignment minimizes
            cost[i, c] = -int(((true == c) & mask).sum())
    row_ind, col_ind = linear_sum_assignment(cost)
    # Build mapping pid -> ground-truth label
    mapping = {pred_ids[r]: col_ind[i] for i, r in enumerate(row_ind)}
    # Unmapped predictions (more clusters than classes) -> map to -1
    pred_remapped = np.array(
        [mapping.get(p, -1) for p in pred], dtype=np.int64)
    return pred_remapped


def micro_f1(true: np.ndarray, pred: np.ndarray) -> float:
    """Micro-averaged F1 for multi-class classification.

    For a single-label classification problem with hard predictions,
    micro-F1 = accuracy. We report it as F1 to match paper terminology.
    """
    if len(true) == 0:
        return 0.0
    tp = int((true == pred).sum())
    return float(tp / len(true))


def structural_fit(X: np.ndarray, labels: np.ndarray,
                   max_dist: float = 10.0) -> float:
    """Structural Org-Fit (SOF): trajectory cohesion around cluster centroid.

    SOF = 1 - mean_distance_to_centroid / max_dist, clipped to [0, 1].
    Higher = trajectories are tightly grouped within their cluster.
    """
    total_dist = 0.0
    count = 0
    for lab in np.unique(labels):
        members = X[labels == lab]
        if len(members) == 0:
            continue
        centroid = members.mean(axis=0)
        for m in members:
            total_dist += float(np.linalg.norm(m - centroid))
            count += 1
    if count == 0:
        return 0.0
    avg = total_dist / count
    return float(max(0.0, 1.0 - avg / max_dist))


def functional_fit(X_obs: np.ndarray, labels: np.ndarray,
                   max_dist: float = 10.0) -> float:
    """Functional Org-Fit (FOF): observation-only cohesion (goals).

    Same formula as SOF but only over the observation portion -- captures
    whether agents in the same role face the same context (goals).
    """
    return structural_fit(X_obs, labels, max_dist=max_dist)


def org_fit(traj_full: np.ndarray, traj_obs: np.ndarray,
            true_labels: np.ndarray, n_classes: int) -> dict:
    """Full Org-Fit evaluation: F1 + SOF + FOF + OF.

    Args:
        traj_full: (N, D_full) full trajectory features (obs+act flattened).
        traj_obs:  (N, D_obs)  observation-only trajectory features.
        true_labels: (N,) ground-truth role labels in [0, n_classes).
        n_classes: number of distinct roles.

    Returns:
        dict with keys: f1, sof, fof, of.
    """
    if len(traj_full) == 0:
        return {"f1": 0.0, "sof": 0.0, "fof": 0.0, "of": 0.0}

    pred = cluster_trajectories(traj_full, n_classes)
    pred_remapped = hungarian_assignment(true_labels, pred, n_classes)
    f1 = micro_f1(true_labels, pred_remapped)
    sof = structural_fit(traj_full, pred)
    fof = functional_fit(traj_obs, pred)
    of = (sof + fof) / 2.0
    return {"f1": float(f1), "sof": float(sof), "fof": float(fof),
            "of": float(of)}
