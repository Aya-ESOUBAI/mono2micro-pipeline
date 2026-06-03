"""
src/clustering/clusterer.py  — OpenBLAS-FREE version
──────────────────────────────────────────────────────
Pure Python clustering. Reads D_similarity.json.
No scipy linkage, no sklearn SpectralClustering (both trigger OpenBLAS).

Algorithms:
  1. HAC  — pure Python Ward-like agglomerative (average linkage)
  2. Louvain — greedy modularity (pure Python)
  3. Spectral-lite — power-iteration on normalised Laplacian + k-means (pure Python)
"""

import csv
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_sim(output_dir):
    path = Path(output_dir) / "D_similarity.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["fqns"], data["matrix"]


def write_labels(path, fqns, labels, col_name):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fqn", col_name])
        for fqn, lbl in zip(fqns, labels):
            w.writerow([fqn, lbl])
    n_clusters = len(set(labels))
    log.info("%s: %d clusters", Path(path).name, n_clusters)
    return n_clusters


def dist_mat(D):
    """Return distance matrix = 1 - similarity, clipped to [0,1]."""
    return [[max(0.0, min(1.0, 1.0 - D[i][j]))
             for j in range(len(D[i]))]
            for i in range(len(D))]


# ─────────────────────────────────────────────────────────────────────────────
# 1. HAC — average linkage (pure Python)
# ─────────────────────────────────────────────────────────────────────────────

def hac_average(dist, k):
    """
    Agglomerative clustering with average linkage.
    dist : N x N list-of-lists (distances).
    k    : target number of clusters.
    Returns list of cluster labels (int, 0-indexed).
    """
    N = len(dist)
    # each node starts in its own cluster
    cluster_of  = list(range(N))          # node -> cluster id
    members     = {i: [i] for i in range(N)}  # cluster id -> node list
    active      = set(range(N))

    linkage_log = []   # (distance, merged_a, merged_b) for dendrogram text

    while len(active) > k:
        # find closest pair of active clusters
        best_d = float("inf")
        best_a = best_b = -1
        active_list = sorted(active)
        for ii, a in enumerate(active_list):
            for b in active_list[ii + 1:]:
                # average linkage: mean of all pairwise distances
                d_sum = 0.0
                count = 0
                for na in members[a]:
                    for nb in members[b]:
                        d_sum += dist[na][nb]
                        count += 1
                avg_d = d_sum / count if count else float("inf")
                if avg_d < best_d:
                    best_d = avg_d
                    best_a, best_b = a, b

        # merge best_b into best_a
        linkage_log.append((round(best_d, 4), best_a, best_b))
        members[best_a].extend(members[best_b])
        for node in members[best_b]:
            cluster_of[node] = best_a
        del members[best_b]
        active.discard(best_b)

    # relabel clusters 0..k-1
    mapping = {cid: i for i, cid in enumerate(sorted(active))}
    labels  = [mapping[cluster_of[n]] for n in range(N)]

    # save text dendrogram
    return labels, linkage_log


def save_text_dendrogram(linkage_log, fqns, path):
    lines = ["HAC Dendrogram (average linkage)\n",
             "Steps are shown bottom-up (last merge = root)\n",
             "=" * 60 + "\n"]
    active_names = {i: fqns[i].split(".")[-1] for i in range(len(fqns))}
    ctr = len(fqns)
    tmp = {i: fqns[i].split(".")[-1] for i in range(len(fqns))}
    for step, (d, a, b) in enumerate(linkage_log, 1):
        na = tmp.get(a, f"cluster_{a}")
        nb = tmp.get(b, f"cluster_{b}")
        new_name = f"[{na} + {nb}]"
        tmp[a]  = new_name
        lines.append(f"Step {step:3d}  dist={d:.4f}  {na}  +  {nb}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log.info("Dendrogram text saved -> %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Louvain — pure Python
# ─────────────────────────────────────────────────────────────────────────────

def louvain(D, max_iter=40):
    N = len(D)
    W = [[D[i][j] if i != j else 0.0 for j in range(N)] for i in range(N)]
    m = sum(W[i][j] for i in range(N) for j in range(N)) / 2 or 1.0
    k = [sum(W[i]) for i in range(N)]

    labels = list(range(N))

    for _ in range(max_iter):
        changed = False
        for i in range(N):
            ci = labels[i]
            comm_delta = defaultdict(float)
            for j in range(N):
                if i == j or W[i][j] == 0:
                    continue
                cj = labels[j]
                if cj == ci:
                    continue
                comm_delta[cj] += W[i][j] - k[i] * k[j] / (2 * m)

            if not comm_delta:
                continue
            best_c    = max(comm_delta, key=comm_delta.get)
            best_gain = comm_delta[best_c]
            # gain of staying in current community
            stay_gain = sum(
                W[i][j] - k[i] * k[labels[j]] / (2 * m)
                for j in range(N) if j != i and labels[j] == ci
            )
            if best_gain > stay_gain + 1e-9:
                labels[i] = best_c
                changed   = True
        if not changed:
            break

    # relabel 0..K-1
    unique  = sorted(set(labels))
    mapping = {v: idx for idx, v in enumerate(unique)}
    return [mapping[l] for l in labels]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spectral-lite — pure Python k-means on top-k eigenvectors
# ─────────────────────────────────────────────────────────────────────────────

def mat_mul(A, B):
    """Multiply two square matrices (list-of-lists)."""
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for kk in range(n):
                s += A[i][kk] * B[kk][j]
            C[i][j] = s
    return C


def power_iteration(A, num_vectors, num_iter=30):
    """
    Find top num_vectors eigenvectors of A via block power iteration.
    A: symmetric n x n (list-of-lists).
    Returns list of n-vectors (each is a list of floats).
    """
    n = len(A)
    random.seed(42)
    # start with random matrix Q (n x num_vectors)
    Q = [[random.gauss(0, 1) for _ in range(num_vectors)] for _ in range(n)]

    for _ in range(num_iter):
        # Z = A @ Q  (n x num_vectors)
        Z = [[0.0] * num_vectors for _ in range(n)]
        for i in range(n):
            for c in range(num_vectors):
                s = 0.0
                for kk in range(n):
                    s += A[i][kk] * Q[kk][c]
                Z[i][c] = s
        # QR decomposition via Gram-Schmidt
        Q_new = []
        for c in range(num_vectors):
            v = [Z[i][c] for i in range(n)]
            for prev in Q_new:
                dot = sum(v[i] * prev[i] for i in range(n))
                v   = [v[i] - dot * prev[i] for i in range(n)]
            norm = math.sqrt(sum(x * x for x in v)) or 1e-10
            Q_new.append([x / norm for x in v])
        Q = [[Q_new[c][i] for c in range(num_vectors)] for i in range(n)]

    # Q is now n x num_vectors; return as list of row vectors
    return Q  # Q[i] = embedding of node i


def kmeans_pure(X, k, max_iter=100):
    """k-means on list-of-vectors X. Returns list of labels."""
    n = len(X)
    dim = len(X[0])
    random.seed(42)
    # init centroids by picking k random points
    centroids = [list(X[i]) for i in random.sample(range(n), min(k, n))]

    labels = [0] * n
    for _ in range(max_iter):
        # assign
        new_labels = []
        for x in X:
            best_c = 0
            best_d = float("inf")
            for ci, c in enumerate(centroids):
                d = sum((x[j] - c[j]) ** 2 for j in range(dim))
                if d < best_d:
                    best_d = d
                    best_c = ci
            new_labels.append(best_c)

        if new_labels == labels:
            break
        labels = new_labels

        # update centroids
        sums   = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for i, lbl in enumerate(labels):
            for j in range(dim):
                sums[lbl][j] += X[i][j]
            counts[lbl] += 1
        for ci in range(k):
            if counts[ci] > 0:
                centroids[ci] = [s / counts[ci] for s in sums[ci]]

    # relabel 0..k-1 (some clusters may be empty)
    unique  = sorted(set(labels))
    mapping = {v: idx for idx, v in enumerate(unique)}
    return [mapping[l] for l in labels]


def spectral_lite(D, k):
    """
    Spectral clustering on similarity matrix D.
    1. Build normalised Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
    2. Power-iterate to get top-k eigenvectors of A (we skip L for simplicity)
    3. k-means on eigenvectors
    """
    N = len(D)
    # degree vector
    deg = [sum(D[i]) for i in range(N)]
    # D^{-1/2}
    d_inv_sqrt = [1.0 / math.sqrt(di) if di > 1e-10 else 0.0 for di in deg]
    # Normalised adjacency A_norm[i][j] = d^{-1/2}[i] * D[i][j] * d^{-1/2}[j]
    A_norm = [
        [d_inv_sqrt[i] * D[i][j] * d_inv_sqrt[j] for j in range(N)]
        for i in range(N)
    ]
    # top-k eigenvectors via power iteration
    vecs = power_iteration(A_norm, num_vectors=min(k, N), num_iter=20)
    # vecs[i] is the embedding for node i (length k)
    labels = kmeans_pure(vecs, k)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Main Clusterer class
# ─────────────────────────────────────────────────────────────────────────────

class Clusterer:
    def __init__(self, output_dir: str, n_clusters: int = 5):
        self.out = Path(output_dir)
        self.res = self.out / "clustering_results"
        self.res.mkdir(parents=True, exist_ok=True)
        self.k   = n_clusters

    def run(self):
        fqns, D = read_sim(self.out)
        N = len(fqns)
        log.info("Loaded similarity matrix: %d x %d", N, N)

        self._run_hac(fqns, D, N)
        self._run_louvain(fqns, D)
        self._run_spectral(fqns, D, N)
        log.info("Clustering complete -> %s", self.res)

    def _run_hac(self, fqns, D, N):
        log.info("Running HAC (average linkage, pure Python) ...")
        d = dist_mat(D)
        labels, linkage_log = hac_average(d, self.k)
        write_labels(self.res / "hac_labels.csv", fqns, labels, "hac_cluster")
        save_text_dendrogram(linkage_log, fqns, self.res / "dendrogram.txt")

    def _run_louvain(self, fqns, D):
        log.info("Running Louvain (pure Python) ...")
        labels = louvain(D)
        write_labels(self.res / "louvain_labels.csv", fqns, labels, "louvain_cluster")

    def _run_spectral(self, fqns, D, N):
        log.info("Running Spectral-lite (pure Python) ...")
        k = min(self.k, N - 1)
        labels = spectral_lite(D, k)
        write_labels(self.res / "spectral_labels.csv", fqns, labels, "spectral_cluster")
