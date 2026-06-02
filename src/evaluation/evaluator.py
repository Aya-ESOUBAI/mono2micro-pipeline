"""
src/evaluation/evaluator.py
────────────────────────────
Computes and reports clustering quality metrics:

  External (vs ground truth from spring-petclinic-microservices):
    ARI, NMI, Purity

  Graph-structure:
    Structural Modularity (SM), Inter-Call Percentage (ICP), Modularity Q

  Internal (no labels needed):
    Silhouette, Davies-Bouldin, Calinski-Harabasz

Writes:
  outputs/clustering_results/evaluation_report.md
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

log = logging.getLogger(__name__)

# ── PetClinic ground-truth (spring-petclinic-microservices mapping) ───────────
# Maps simple class name patterns -> target microservice
# Source: github.com/spring-petclinic/spring-petclinic-microservices
BUILTIN_GROUND_TRUTH = {
    # customers-service
    "Owner":           "customers-service",
    "Pet":             "customers-service",
    "PetType":         "customers-service",
    "OwnerRepository": "customers-service",
    "PetRepository":   "customers-service",
    "OwnerController": "customers-service",
    "PetController":   "customers-service",
    "OwnerValidator":  "customers-service",
    "PetValidator":    "customers-service",
    # vets-service
    "Vet":             "vets-service",
    "Specialty":       "vets-service",
    "VetRepository":   "vets-service",
    "VetController":   "vets-service",
    # visits-service
    "Visit":           "visits-service",
    "VisitRepository": "visits-service",
    "VisitController": "visits-service",
    # api-gateway / shared
    "CrashController": "api-gateway",
    "WelcomeController":"api-gateway",
    # config / infra
    "PetClinicApplication": "config-server",
    "CacheConfig":          "config-server",
    "MvcConfig":            "config-server",
    "BaseEntity":           "config-server",
    "NamedEntity":          "config-server",
    "ClinicService":        "customers-service",
}

SERVICE_TO_ID = {
    "customers-service": 0,
    "vets-service":      1,
    "visits-service":    2,
    "api-gateway":       3,
    "config-server":     4,
}


def purity(labels_true, labels_pred) -> float:
    from collections import Counter
    n = len(labels_true)
    clusters: dict = {}
    for t, p in zip(labels_true, labels_pred):
        clusters.setdefault(p, []).append(t)
    return sum(Counter(v).most_common(1)[0][1] for v in clusters.values()) / n


def modularity_q(D: np.ndarray, labels: np.ndarray) -> float:
    """
    Newman-Girvan modularity Q on the similarity-as-adjacency matrix.
    Q = (1/2m) Σ_{ij} [A_ij - k_i*k_j/(2m)] δ(c_i, c_j)
    """
    A = D.copy()
    np.fill_diagonal(A, 0)
    m = A.sum() / 2
    if m == 0:
        return 0.0
    k = A.sum(axis=1)
    Q = 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == labels[j]:
                Q += A[i, j] - k[i] * k[j] / (2 * m)
    return Q / (2 * m)


def structural_modularity(edges: list[dict], labels: dict[str, int]) -> tuple[float, float]:
    """
    SM (Structural Modularity): fraction of intra-cluster calls.
    ICP (Inter-Call Percentage): fraction of inter-cluster calls.
    """
    intra = inter = 0
    for e in edges:
        ci = labels.get(e["fromClass"], -1)
        cj = labels.get(e["toClass"],   -1)
        if ci < 0 or cj < 0:
            continue
        if ci == cj:
            intra += 1
        else:
            inter += 1
    total = intra + inter or 1
    return intra / total, inter / total


class Evaluator:
    def __init__(self, output_dir: str, ground_truth_csv: str | None = None):
        self.out = Path(output_dir)
        self.res = self.out / "clustering_results"
        self.gt_csv = Path(ground_truth_csv) if ground_truth_csv else None

    def run(self):
        # ── load artefacts ─────────────────────────────────────────────────────
        data  = np.load(self.out / "D_similarity.npz", allow_pickle=True)
        D     = data["sim"]
        fqns  = list(data["fqns"])
        X_df  = pd.read_csv(self.out / "X_features.csv")
        cg    = json.loads((self.out / "call_graph.json").read_text())

        num_cols = [c for c in X_df.columns
                    if c not in {"fqn", "louvain_community"}]
        X = X_df[num_cols].values

        # ── load / build ground truth ──────────────────────────────────────────
        gt_labels = self._build_ground_truth(fqns)

        # ── algorithm result files ─────────────────────────────────────────────
        algos = {
            "HAC":      ("hac_labels.csv",      "hac_cluster"),
            "Louvain":  ("louvain_labels.csv",   "louvain_cluster"),
            "Spectral": ("spectral_labels.csv",  "spectral_cluster"),
        }

        report_sections = []
        for algo, (fname, col) in algos.items():
            path = self.res / fname
            if not path.exists():
                log.warning("Missing: %s — skipping", fname)
                continue
            df = pd.read_csv(path)
            label_map = dict(zip(df["fqn"], df[col]))
            pred = np.array([label_map.get(fqn, 0) for fqn in fqns])
            section = self._evaluate_one(algo, pred, gt_labels, X, D, cg["edges"],
                                         {fqn: int(p) for fqn, p in zip(fqns, pred)})
            report_sections.append(section)

        self._write_report(report_sections)

    def _build_ground_truth(self, fqns: list[str]) -> np.ndarray:
        if self.gt_csv and self.gt_csv.exists():
            gt_df = pd.read_csv(self.gt_csv)
            gt_map = dict(zip(gt_df["fqn"], gt_df["service"]))
        else:
            gt_map = {}

        labels = []
        for fqn in fqns:
            simple = fqn.split(".")[-1]
            svc    = gt_map.get(fqn) or BUILTIN_GROUND_TRUTH.get(simple, "unknown")
            labels.append(SERVICE_TO_ID.get(svc, 5))
        return np.array(labels)

    def _evaluate_one(
        self,
        algo: str,
        pred: np.ndarray,
        gt: np.ndarray,
        X: np.ndarray,
        D: np.ndarray,
        edges: list[dict],
        label_map: dict[str, int],
    ) -> str:
        n_clusters = len(set(pred))
        lines = [f"## {algo}  ({n_clusters} clusters)\n"]

        # external
        if len(set(gt)) > 1:
            ari  = adjusted_rand_score(gt, pred)
            nmi  = normalized_mutual_info_score(gt, pred)
            pur  = purity(gt, pred)
            lines.append("### External (vs spring-petclinic-microservices ground truth)")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| ARI    | {ari:.4f} |")
            lines.append(f"| NMI    | {nmi:.4f} |")
            lines.append(f"| Purity | {pur:.4f} |")
            lines.append("")

        # internal
        if n_clusters >= 2 and X.shape[0] > n_clusters:
            sil = silhouette_score(X, pred, metric="euclidean")
            db  = davies_bouldin_score(X, pred)
            ch  = calinski_harabasz_score(X, pred)
        else:
            sil = db = ch = float("nan")

        lines.append("### Internal metrics")
        lines.append("| Metric              | Value |")
        lines.append("|---------------------|-------|")
        lines.append(f"| Silhouette          | {sil:.4f} |")
        lines.append(f"| Davies-Bouldin      | {db:.4f} |")
        lines.append(f"| Calinski-Harabasz   | {ch:.2f} |")
        lines.append("")

        # graph-structure
        sm, icp = structural_modularity(edges, label_map)
        q = modularity_q(D, pred)
        lines.append("### Graph-structure metrics")
        lines.append("| Metric                           | Value |")
        lines.append("|----------------------------------|-------|")
        lines.append(f"| Structural Modularity (SM)       | {sm:.4f} |")
        lines.append(f"| Inter-Call Percentage (ICP)      | {icp:.4f} |")
        lines.append(f"| Modularity Q                     | {q:.4f} |")
        lines.append("")

        return "\n".join(lines)

    def _write_report(self, sections: list[str]):
        header = [
            "# Evaluation Report — Mono2Micro Clustering",
            "",
            "> Generated by `mono2micro-pipeline/src/evaluation/evaluator.py`",
            "",
            "Ground truth: [spring-petclinic-microservices]"
            "(https://github.com/spring-petclinic/spring-petclinic-microservices)",
            "",
            "Algorithms compared: **HAC** · **Louvain** · **Spectral**",
            "",
            "---",
            "",
        ]
        body = "\n\n---\n\n".join(sections)
        footer = [
            "",
            "---",
            "",
            "## Interpretation Guide",
            "",
            "| Metric | Better when |",
            "|--------|-------------|",
            "| ARI    | -> 1.0       |",
            "| NMI    | -> 1.0       |",
            "| Purity | -> 1.0       |",
            "| Silhouette | -> 1.0   |",
            "| Davies-Bouldin | -> 0  |",
            "| Calinski-Harabasz | higher |",
            "| SM (intra-call %) | -> 1.0 |",
            "| ICP (inter-call %) | -> 0.0 |",
            "| Modularity Q | -> 1.0 |",
        ]
        report = "\n".join(header) + "\n" + body + "\n".join(footer) + "\n"
        out_path = self.res / "evaluation_report.md"
        out_path.write_text(report, encoding="utf-8")
        log.info("Evaluation report -> %s", out_path)
