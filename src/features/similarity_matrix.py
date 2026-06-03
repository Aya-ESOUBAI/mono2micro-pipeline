"""
src/features/similarity_matrix.py  — OpenBLAS-FREE version
────────────────────────────────────────────────────────────
Pure Python/CSV. No numpy matrix ops, no scipy, no OpenBLAS.
Writes D_similarity.csv (N x N) and D_similarity.json (for clusterer).
"""

import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"alpha": 0.35, "beta": 0.30, "gamma": 0.20, "delta": 0.15}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def cosine_dicts(u, v):
    """Cosine similarity between two sparse dicts {key: float}."""
    dot  = sum(u.get(k, 0) * v.get(k, 0) for k in u)
    nu   = math.sqrt(sum(x * x for x in u.values()))
    nv   = math.sqrt(sum(x * x for x in v.values()))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def read_csv_as_dicts(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class SimilarityMatrixBuilder:
    def __init__(self, output_dir: str, weights=None):
        self.out = Path(output_dir)
        w = weights or DEFAULT_WEIGHTS
        total = sum(w.values())
        self.w = {k: v / total for k, v in w.items()}

    def run(self):
        cg = json.loads((self.out / "call_graph.json").read_text(encoding="utf-8"))
        db = json.loads((self.out / "db_features.json").read_text(encoding="utf-8"))

        # Read X_features.csv as plain dicts
        xf_rows = read_csv_as_dicts(self.out / "X_features.csv")
        fqns = [r["fqn"] for r in xf_rows]
        N    = len(fqns)
        idx  = {fqn: i for i, fqn in enumerate(fqns)}
        log.info("Building %d x %d similarity matrix (pure Python)...", N, N)

        # callees per class
        callees = defaultdict(set)
        for e in cg["edges"]:
            if e["fromClass"] in idx:
                callees[e["fromClass"]].add(e["toClass"])

        # table sets
        table_sets = defaultdict(set)
        for entry in db["classes"]:
            if entry["fqn"] in idx:
                table_sets[entry["fqn"]] = set(entry["tables"])

        # TF-IDF vectors as sparse dicts
        tfidf_vecs = {}
        for r in xf_rows:
            fqn = r["fqn"]
            vec = {k: float(v) for k, v in r.items()
                   if k.startswith("tfidf_") and v not in ("", "0", "0.0")}
            tfidf_vecs[fqn] = vec

        # behavioral co-occurrence
        log_cooc = defaultdict(float)
        logs_path = self.out / "logs_features.json"
        if logs_path.exists():
            logs = json.loads(logs_path.read_text(encoding="utf-8"))
            traces = logs.get("traces", [])
            if traces:
                log.info("Using %d traces for behavioral sim", len(traces))
                raw_cooc = defaultdict(float)
                for trace in traces:
                    cls_in_trace = [c for c in trace.get("classes", []) if c in idx]
                    for ci in cls_in_trace:
                        for cj in cls_in_trace:
                            if ci != cj:
                                raw_cooc[(ci, cj)] += 1
                max_c = max(raw_cooc.values()) if raw_cooc else 1
                for (ci, cj), v in raw_cooc.items():
                    log_cooc[(ci, cj)] = v / max_c

        alpha = self.w["alpha"]
        beta  = self.w["beta"]
        gamma = self.w["gamma"]
        delta = self.w["delta"]

        # Build matrix row by row — store as list of lists of floats
        D = []
        for i in range(N):
            row = []
            fi  = fqns[i]
            pkg_i = ".".join(fi.split(".")[:-1])
            for j in range(N):
                if i == j:
                    row.append(1.0)
                    continue
                fj    = fqns[j]
                s_str = jaccard(callees[fi], callees[fj])
                s_dat = jaccard(table_sets[fi], table_sets[fj])
                s_sem = cosine_dicts(tfidf_vecs.get(fi, {}), tfidf_vecs.get(fj, {}))
                s_beh = log_cooc.get((fi, fj), 0.0)
                sim   = alpha * s_str + beta * s_dat + gamma * s_sem + delta * s_beh
                # same-package bonus
                pkg_j = ".".join(fj.split(".")[:-1])
                if pkg_i == pkg_j:
                    sim = min(1.0, sim + 0.05)
                row.append(round(sim, 6))
            D.append(row)

        # Save as JSON (lightweight, no numpy needed in clusterer)
        sim_data = {"fqns": fqns, "matrix": D}
        (self.out / "D_similarity.json").write_text(
            json.dumps(sim_data), encoding="utf-8"
        )
        log.info("D_similarity.json saved (%d x %d)", N, N)

        # Also save as CSV for inspection
        with open(self.out / "D_similarity.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fqn"] + fqns)
            for i, fqn in enumerate(fqns):
                w.writerow([fqn] + [f"{v:.4f}" for v in D[i]])
        log.info("D_similarity.csv saved")

        log.info("Weights: a(struct)=%.2f b(data)=%.2f g(semantic)=%.2f d(behavioral)=%.2f",
                 alpha, beta, gamma, delta)
