"""
src/features/similarity_matrix.py
──────────────────────────────────
Builds D_similarity: an N×N hybrid similarity matrix

  sim(i,j) = α·struct_sim + β·data_sim + γ·semantic_sim + δ·behavioral_sim

Each component in [0, 1]:
  • struct_sim  — Jaccard on {callee set} ∪ neighbour normalisation
  • data_sim    — Jaccard on table-access sets
  • semantic_sim — cosine on TF-IDF vectors
  • behavioral_sim — co-occurrence in request traces (from logs_features.json if present)

Outputs:
  outputs/D_similarity.npz   (scipy sparse, key='sim')
  outputs/D_similarity.csv   (dense, for inspection / small N)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"alpha": 0.35, "beta": 0.30, "gamma": 0.20, "delta": 0.15}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


class SimilarityMatrixBuilder:
    def __init__(self, output_dir: str, weights: dict | None = None):
        self.out = Path(output_dir)
        self.w   = weights or DEFAULT_WEIGHTS
        # normalise weights
        total = sum(self.w.values())
        self.w = {k: v / total for k, v in self.w.items()}

    def run(self):
        cg  = json.loads((self.out / "call_graph.json").read_text())
        db  = json.loads((self.out / "db_features.json").read_text())
        xf  = pd.read_csv(self.out / "X_features.csv")

        fqns = list(xf["fqn"])
        N    = len(fqns)
        idx  = {fqn: i for i, fqn in enumerate(fqns)}
        log.info("Building %d × %d similarity matrix …", N, N)

        # ── callees per class (for structural sim) ─────────────────────────────
        callees: dict[str, set[str]] = {fqn: set() for fqn in fqns}
        for e in cg["edges"]:
            if e["fromClass"] in callees:
                callees[e["fromClass"]].add(e["toClass"])

        # ── table sets per class (data sim) ───────────────────────────────────
        table_sets: dict[str, set[str]] = {fqn: set() for fqn in fqns}
        for entry in db["classes"]:
            fqn = entry["fqn"]
            if fqn in table_sets:
                table_sets[fqn] = set(entry["tables"])

        # ── TF-IDF vectors (semantic sim) — use tfidf_* columns ───────────────
        tfidf_cols = [c for c in xf.columns if c.startswith("tfidf_")]
        tfidf_mat  = xf[tfidf_cols].values if tfidf_cols else np.zeros((N, 1))

        # ── behavioral: load logs if available ────────────────────────────────
        log_cooc = np.zeros((N, N))
        logs_path = self.out / "logs_features.json"
        if logs_path.exists():
            logs = json.loads(logs_path.read_text())
            traces = logs.get("traces", [])
            if traces:
                log.info("Using %d real request traces for behavioral sim", len(traces))
                for trace in traces:
                    classes_in_trace = [c for c in trace.get("classes", []) if c in idx]
                    for ci in classes_in_trace:
                        for cj in classes_in_trace:
                            if ci != cj:
                                log_cooc[idx[ci], idx[cj]] += 1
                # normalise to [0,1]
                max_cooc = log_cooc.max()
                if max_cooc > 0:
                    log_cooc /= max_cooc
            else:
                log.warning("logs_features.json has no 'traces' key — behavioral sim = 0")

        # ── compute similarity ─────────────────────────────────────────────────
        D = np.zeros((N, N))
        alpha = self.w["alpha"]
        beta  = self.w["beta"]
        gamma = self.w["gamma"]
        delta = self.w["delta"]

        for i in range(N):
            for j in range(i, N):
                fi, fj = fqns[i], fqns[j]
                if i == j:
                    D[i, j] = 1.0
                    continue

                s_struct    = jaccard(callees[fi], callees[fj])
                s_data      = jaccard(table_sets[fi], table_sets[fj])
                s_semantic  = cosine(tfidf_mat[i], tfidf_mat[j])
                s_behavioral= float(log_cooc[i, j])

                sim = (alpha * s_struct +
                       beta  * s_data +
                       gamma * s_semantic +
                       delta * s_behavioral)

                # small same-package bonus (+0.05) to encourage cohesion
                pkg_i = ".".join(fi.split(".")[:-1])
                pkg_j = ".".join(fj.split(".")[:-1])
                if pkg_i == pkg_j:
                    sim = min(1.0, sim + 0.05)

                D[i, j] = D[j, i] = sim

        # ── save ──────────────────────────────────────────────────────────────
        np.savez_compressed(self.out / "D_similarity.npz", sim=D, fqns=np.array(fqns))
        log.info("D_similarity.npz saved (%d × %d)", N, N)

        sim_df = pd.DataFrame(D, index=fqns, columns=fqns)
        sim_df.to_csv(self.out / "D_similarity.csv")
        log.info("D_similarity.csv saved")

        # ── weight report ──────────────────────────────────────────────────────
        log.info(
            "Weights: α(struct)=%.2f β(data)=%.2f γ(semantic)=%.2f δ(behavioral)=%.2f",
            alpha, beta, gamma, delta,
        )
