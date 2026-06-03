"""
src/evaluation/evaluator.py  — OpenBLAS-FREE version
──────────────────────────────────────────────────────
Pure Python metrics. No sklearn, no numpy, no scipy.
Reads CSV files directly.
"""

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# Ground truth from spring-petclinic-microservices
BUILTIN_GROUND_TRUTH = {
    "Owner":           "customers-service",
    "Pet":             "customers-service",
    "PetType":         "customers-service",
    "OwnerRepository": "customers-service",
    "PetRepository":   "customers-service",
    "OwnerController": "customers-service",
    "PetController":   "customers-service",
    "OwnerValidator":  "customers-service",
    "PetValidator":    "customers-service",
    "Vet":             "vets-service",
    "Specialty":       "vets-service",
    "VetRepository":   "vets-service",
    "VetController":   "vets-service",
    "Visit":           "visits-service",
    "VisitRepository": "visits-service",
    "VisitController": "visits-service",
    "CrashController": "api-gateway",
    "WelcomeController": "api-gateway",
    "PetClinicApplication": "config-server",
    "CacheConfig":     "config-server",
    "MvcConfig":       "config-server",
    "BaseEntity":      "config-server",
    "NamedEntity":     "config-server",
    "ClinicService":   "customers-service",
}

SERVICE_TO_ID = {
    "customers-service": 0,
    "vets-service":      1,
    "visits-service":    2,
    "api-gateway":       3,
    "config-server":     4,
}


# ── Pure Python metrics ───────────────────────────────────────────────────────

def purity(y_true, y_pred):
    clusters = defaultdict(list)
    for t, p in zip(y_true, y_pred):
        clusters[p].append(t)
    correct = sum(Counter(v).most_common(1)[0][1] for v in clusters.values())
    return correct / len(y_true) if y_true else 0.0


def entropy(labels):
    counts = Counter(labels)
    total  = len(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def nmi(y_true, y_pred):
    """Normalized Mutual Information (pure Python)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    # contingency
    cont = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        cont[(t, p)] += 1
    ct = Counter(y_true)
    cp = Counter(y_pred)
    mi = 0.0
    for (t, p), cnt in cont.items():
        if cnt > 0:
            mi += (cnt / n) * math.log2((cnt * n) / (ct[t] * cp[p]))
    h_true = entropy(y_true)
    h_pred = entropy(y_pred)
    denom  = (h_true + h_pred) / 2
    return mi / denom if denom > 0 else 0.0


def ari(y_true, y_pred):
    """Adjusted Rand Index (pure Python)."""
    n = len(y_true)
    # contingency matrix counts
    cont = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        cont[(t, p)] += 1
    # sum of C(n_ij, 2)
    sum_cij = sum(v * (v - 1) // 2 for v in cont.values())
    ct = Counter(y_true)
    cp = Counter(y_pred)
    sum_ci = sum(v * (v - 1) // 2 for v in ct.values())
    sum_cj = sum(v * (v - 1) // 2 for v in cp.values())
    cn2 = n * (n - 1) // 2
    expected = sum_ci * sum_cj / cn2 if cn2 > 0 else 0
    max_index = (sum_ci + sum_cj) / 2
    denom = max_index - expected
    return (sum_cij - expected) / denom if denom > 0 else 0.0


def silhouette(X_rows, labels):
    """
    X_rows : list of dicts (features).
    labels : list of int cluster labels.
    Pure Python silhouette score.
    """
    fqns   = [r["fqn"] for r in X_rows]
    num_cols = [c for c in X_rows[0] if c not in {"fqn", "louvain_community"}]

    def euclidean(a, b):
        return math.sqrt(sum((float(a.get(c, 0)) - float(b.get(c, 0))) ** 2
                             for c in num_cols))

    n = len(X_rows)
    s_scores = []
    for i in range(n):
        ci = labels[i]
        # intra-cluster distances
        intra = [euclidean(X_rows[i], X_rows[j])
                 for j in range(n) if j != i and labels[j] == ci]
        if not intra:
            s_scores.append(0.0)
            continue
        a = sum(intra) / len(intra)
        # nearest other cluster
        other_clusters = set(labels) - {ci}
        if not other_clusters:
            s_scores.append(0.0)
            continue
        b = min(
            sum(euclidean(X_rows[i], X_rows[j])
                for j in range(n) if labels[j] == ck) /
            max(1, sum(1 for j in range(n) if labels[j] == ck))
            for ck in other_clusters
        )
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        s_scores.append(s)
    return sum(s_scores) / len(s_scores) if s_scores else 0.0


def structural_modularity(edges, label_map):
    intra = inter = 0
    for e in edges:
        ci = label_map.get(e["fromClass"], -1)
        cj = label_map.get(e["toClass"],   -1)
        if ci < 0 or cj < 0:
            continue
        if ci == cj:
            intra += 1
        else:
            inter += 1
    total = intra + inter or 1
    return intra / total, inter / total


def modularity_q(D, labels):
    N = len(labels)
    # D is list-of-lists
    total_w = sum(D[i][j] for i in range(N) for j in range(N) if i != j) / 2
    if total_w == 0:
        return 0.0
    k = [sum(D[i][j] for j in range(N) if j != i) for i in range(N)]
    Q = 0.0
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                Q += D[i][j] - k[i] * k[j] / (2 * total_w)
    return Q / (2 * total_w)


# ─────────────────────────────────────────────────────────────────────────────

def read_csv_dicts(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Evaluator:
    def __init__(self, output_dir: str, ground_truth_csv=None):
        self.out    = Path(output_dir)
        self.res    = self.out / "clustering_results"
        self.gt_csv = Path(ground_truth_csv) if ground_truth_csv else None

    def run(self):
        sim_data = json.loads((self.out / "D_similarity.json").read_text(encoding="utf-8"))
        fqns = sim_data["fqns"]
        D    = sim_data["matrix"]
        N    = len(fqns)

        xf_rows = read_csv_dicts(self.out / "X_features.csv")
        cg      = json.loads((self.out / "call_graph.json").read_text(encoding="utf-8"))

        gt_labels = self._build_gt(fqns)

        algos = [
            ("HAC",      self.res / "hac_labels.csv",      "hac_cluster"),
            ("Louvain",  self.res / "louvain_labels.csv",  "louvain_cluster"),
            ("Spectral", self.res / "spectral_labels.csv", "spectral_cluster"),
        ]

        sections = []
        for algo, path, col in algos:
            if not path.exists():
                log.warning("Missing: %s", path)
                continue
            rows     = read_csv_dicts(path)
            lbl_map  = {r["fqn"]: int(r[col]) for r in rows}
            pred     = [lbl_map.get(fqn, 0) for fqn in fqns]
            sections.append(self._evaluate(algo, pred, gt_labels, xf_rows, D,
                                            cg["edges"], lbl_map))

        report = self._build_report(sections)
        out_path = self.res / "evaluation_report.md"
        out_path.write_text(report, encoding="utf-8")
        log.info("Evaluation report -> %s", out_path)

    def _build_gt(self, fqns):
        gt_map = {}
        if self.gt_csv and self.gt_csv.exists():
            for r in read_csv_dicts(self.gt_csv):
                gt_map[r["fqn"]] = r["service"]
        labels = []
        for fqn in fqns:
            simple = fqn.split(".")[-1]
            svc    = gt_map.get(fqn) or BUILTIN_GROUND_TRUTH.get(simple, "unknown")
            labels.append(SERVICE_TO_ID.get(svc, 5))
        return labels

    def _evaluate(self, algo, pred, gt, xf_rows, D, edges, lbl_map):
        k = len(set(pred))
        lines = [f"## {algo}  ({k} clusters)\n"]

        # External
        if len(set(gt)) > 1:
            _ari  = ari(gt, pred)
            _nmi  = nmi(gt, pred)
            _pur  = purity(gt, pred)
            lines += [
                "### External (vs spring-petclinic-microservices)",
                "| Metric | Value |", "|--------|-------|",
                f"| ARI    | {_ari:.4f} |",
                f"| NMI    | {_nmi:.4f} |",
                f"| Purity | {_pur:.4f} |", "",
            ]

        # Internal (silhouette only — pure Python, skipping DB/CH for speed)
        if k >= 2 and len(xf_rows) > k:
            _sil = silhouette(xf_rows, pred)
        else:
            _sil = float("nan")
        lines += [
            "### Internal metrics",
            "| Metric     | Value |", "|------------|-------|",
            f"| Silhouette | {_sil:.4f} |", "",
        ]

        # Graph-structure
        sm, icp = structural_modularity(edges, lbl_map)
        q = modularity_q(D, pred)
        lines += [
            "### Graph-structure metrics",
            "| Metric                      | Value |",
            "|-----------------------------|-------|",
            f"| Structural Modularity (SM)  | {sm:.4f} |",
            f"| Inter-Call Percentage (ICP) | {icp:.4f} |",
            f"| Modularity Q                | {q:.4f} |", "",
        ]
        return "\n".join(lines)

    def _build_report(self, sections):
        header = [
            "# Evaluation Report — Mono2Micro Clustering", "",
            "> Generated by mono2micro-pipeline", "",
            "Ground truth: spring-petclinic-microservices", "",
            "Algorithms: **HAC** - **Louvain** - **Spectral**", "",
            "---", "",
        ]
        footer = [
            "", "---", "", "## Metric guide",
            "| Metric | Better when |", "|--------|-------------|",
            "| ARI    | -> 1.0      |", "| NMI    | -> 1.0      |",
            "| Purity | -> 1.0      |", "| Silhouette | -> 1.0  |",
            "| SM (intra-call%) | -> 1.0  |",
            "| ICP (inter-call%) | -> 0.0 |",
            "| Modularity Q | -> 1.0      |", "",
        ]
        return "\n".join(header) + "\n\n---\n\n".join(sections) + "\n".join(footer)
