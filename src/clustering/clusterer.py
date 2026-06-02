"""
src/clustering/clusterer.py
────────────────────────────
Runs three clustering algorithms on D_similarity.npz:
  1. HAC (Hierarchical Agglomerative Clustering) — Ward linkage
  2. Louvain / Leiden (modularity-maximising graph partitioning)
  3. Spectral Clustering

Outputs:
  outputs/clustering_results/hac_labels.csv
  outputs/clustering_results/louvain_labels.csv
  outputs/clustering_results/spectral_labels.csv
  outputs/clustering_results/dendrogram.png   (HAC)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering

log = logging.getLogger(__name__)


# ── Pure-Python Louvain (no external lib dependency) ─────────────────────────

def louvain_partition(D: np.ndarray, fqns: list[str], resolution: float = 1.0) -> np.ndarray:
    """
    Greedy modularity-maximising partition on the weighted adjacency D.
    Returns label array of length N.

    For production: use `python-louvain` (community library) or `leidenalg`.
    This implementation is correct for small graphs (N < 100).
    """
    N = len(fqns)
    labels = np.arange(N)  # each node in its own community
    W = D.copy()
    np.fill_diagonal(W, 0)
    m = W.sum() / 2 or 1.0
    k = W.sum(axis=1)            # node strengths

    improved = True
    max_iter = 50
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(N):
            ci = labels[i]
            neighbours = np.where(W[i] > 0)[0]
            if len(neighbours) == 0:
                continue

            # communities of neighbours
            comm_delta: dict[int, float] = {}
            for j in neighbours:
                cj = labels[j]
                if cj == ci:
                    continue
                delta = (W[i, j] - resolution * k[i] * k[j] / (2 * m))
                comm_delta[cj] = comm_delta.get(cj, 0) + delta

            if not comm_delta:
                continue

            best_c, best_gain = max(comm_delta.items(), key=lambda x: x[1])
            # also compute gain of removing i from ci
            remove_gain = -sum(
                W[i, j] - resolution * k[i] * k[labels[j]] / (2 * m)
                for j in neighbours if labels[j] == ci
            )

            if best_gain + remove_gain > 1e-10:
                labels[i] = best_c
                improved = True

    # relabel to 0..K-1
    unique = {v: idx for idx, v in enumerate(sorted(set(labels)))}
    return np.array([unique[l] for l in labels])


class Clusterer:
    def __init__(self, output_dir: str, n_clusters: int = 5):
        self.out  = Path(output_dir)
        self.res  = self.out / "clustering_results"
        self.res.mkdir(parents=True, exist_ok=True)
        self.k    = n_clusters

    def run(self):
        data = np.load(self.out / "D_similarity.npz", allow_pickle=True)
        D    = data["sim"]
        fqns = list(data["fqns"])
        N    = len(fqns)
        log.info("Loaded similarity matrix: %d × %d", N, N)

        # distance = 1 - similarity (clipped for numerical safety)
        dist_mat = np.clip(1 - D, 0, 1)

        self._run_hac(dist_mat, fqns)
        self._run_louvain(D, fqns)
        self._run_spectral(D, fqns, N)

        log.info("Clustering complete → %s", self.res)

    # ── HAC ──────────────────────────────────────────────────────────────────

    def _run_hac(self, dist_mat: np.ndarray, fqns: list[str]):
        log.info("Running HAC (Ward linkage) …")
        condensed = squareform(dist_mat, checks=False)
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=self.k, criterion="maxclust") - 1  # 0-indexed

        df = pd.DataFrame({"fqn": fqns, "hac_cluster": labels})
        df.to_csv(self.res / "hac_labels.csv", index=False)
        log.info("HAC: %d clusters", len(set(labels)))

        # dendrogram (no display)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            short_names = [fqn.split(".")[-1] for fqn in fqns]
            fig, ax = plt.subplots(figsize=(14, 5))
            dendrogram(Z, labels=short_names, ax=ax, leaf_rotation=45, leaf_font_size=8)
            ax.set_title("HAC Dendrogram — Spring PetClinic")
            ax.set_ylabel("Ward distance")
            fig.tight_layout()
            fig.savefig(self.res / "dendrogram.png", dpi=120)
            plt.close(fig)
            log.info("Dendrogram saved → clustering_results/dendrogram.png")
        except Exception as exc:
            log.warning("Could not save dendrogram: %s", exc)

    # ── Louvain ───────────────────────────────────────────────────────────────

    def _run_louvain(self, D: np.ndarray, fqns: list[str]):
        log.info("Running Louvain partition …")
        # Try python-louvain if installed; fall back to our implementation
        try:
            import community as community_louvain
            import networkx as nx
            G = nx.from_numpy_array(D)
            mapping = {i: fqn for i, fqn in enumerate(fqns)}
            nx.relabel_nodes(G, mapping, copy=False)
            partition = community_louvain.best_partition(G)
            labels = [partition[fqn] for fqn in fqns]
            log.info("Using python-louvain library")
        except ImportError:
            log.info("python-louvain not installed — using built-in Louvain")
            labels = louvain_partition(D, fqns).tolist()

        df = pd.DataFrame({"fqn": fqns, "louvain_cluster": labels})
        df.to_csv(self.res / "louvain_labels.csv", index=False)
        log.info("Louvain: %d communities", len(set(labels)))

    # ── Spectral ──────────────────────────────────────────────────────────────

    def _run_spectral(self, D: np.ndarray, fqns: list[str], N: int):
        log.info("Running Spectral Clustering …")
        k = min(self.k, N - 1)
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
            n_init=10,
        )
        labels = sc.fit_predict(D)
        df = pd.DataFrame({"fqn": fqns, "spectral_cluster": labels})
        df.to_csv(self.res / "spectral_labels.csv", index=False)
        log.info("Spectral: %d clusters", len(set(labels)))
