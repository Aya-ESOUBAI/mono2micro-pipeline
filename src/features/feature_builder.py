"""
src/features/feature_builder.py  — OpenBLAS-FREE version
──────────────────────────────────────────────────────────
Zero NumPy / SciPy matrix ops. Uses only:
  - Python stdlib (csv, math, collections)
  - pandas  (read/write CSV only — no matrix math)
All heavy computation done with plain Python dicts and lists.
"""

import csv
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

TABLES = ["owners", "pets", "visits", "vets", "specialties",
          "vet_specialties", "types", "pet_types"]


# ── camelCase tokeniser ───────────────────────────────────────────────────────
def camel_split(token: str) -> list:
    parts = re.sub(r'([A-Z][a-z]+)', r' \1', token).split()
    return [p.lower() for p in parts if len(p) > 1]


def tokenize_class(methods, javadoc, literals):
    tokens = []
    for m in methods:
        tokens.extend(camel_split(m))
    tokens.extend(camel_split(str(javadoc)))
    for s in literals:
        tokens.extend(str(s).lower().split())
    return tokens


# ── Pure-Python graph metrics (no numpy) ─────────────────────────────────────
def graph_metrics(nodes, edges):
    fqns  = [n["fqn"] for n in nodes]
    idx   = {fqn: i for i, fqn in enumerate(fqns)}
    N     = len(fqns)

    # adjacency
    out_adj = defaultdict(set)
    in_adj  = defaultdict(set)
    for e in edges:
        fi = idx.get(e["fromClass"], -1)
        ti = idx.get(e["toClass"],   -1)
        if fi >= 0 and ti >= 0:
            out_adj[fi].add(ti)
            in_adj[ti].add(fi)

    # PageRank — pure Python dict, no matrix
    pr = {i: 1.0 / N for i in range(N)}
    d  = 0.85
    for _ in range(20):
        new_pr = {}
        for i in range(N):
            rank = (1 - d) / N
            for j in in_adj[i]:
                out_deg = len(out_adj[j]) or 1
                rank += d * pr[j] / out_deg
            new_pr[i] = rank
        pr = new_pr

    # Local clustering coefficient — undirected
    undirected = defaultdict(set)
    for i, js in out_adj.items():
        for j in js:
            undirected[i].add(j)
            undirected[j].add(i)

    cc = {}
    for i in range(N):
        nb = undirected[i]
        k  = len(nb)
        if k < 2:
            cc[i] = 0.0
        else:
            links = sum(1 for u in nb for v in nb if u != v and v in undirected[u])
            cc[i] = links / (k * (k - 1))

    # Approx betweenness via BFS (pure Python)
    from collections import deque
    betw = [0.0] * N

    for s in range(N):
        dist  = [-1] * N
        paths = [0]  * N
        dist[s] = 0
        paths[s] = 1
        q     = deque([s])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in out_adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
                if dist[v] == dist[u] + 1:
                    paths[v] += paths[u]
        dep = [0.0] * N
        for w in reversed(order):
            for v in out_adj[w]:
                if dist[v] == dist[w] + 1 and paths[v] > 0:
                    dep[w] += (paths[w] / paths[v]) * (1 + dep[v])
        for w in range(N):
            if w != s:
                betw[w] += dep[w]

    norm = (N - 1) * (N - 2) or 1
    betw = [b / norm for b in betw]

    # Simple greedy Louvain community
    community = list(range(N))
    for _ in range(30):
        changed = False
        for i in range(N):
            nb = list(out_adj[i] | in_adj[i])
            if not nb:
                continue
            comm_cnt = defaultdict(int)
            for j in nb:
                comm_cnt[community[j]] += 1
            best_c    = community[i]
            best_gain = 0
            for c, cnt in comm_cnt.items():
                gain = cnt - comm_cnt.get(community[i], 0)
                if gain > best_gain:
                    best_gain = gain
                    best_c    = c
            if best_c != community[i]:
                community[i] = best_c
                changed = True
        if not changed:
            break

    return {
        fqns[i]: {
            "pagerank":         round(pr[i], 6),
            "clustering_coeff": round(cc[i], 4),
            "betweenness":      round(betw[i], 6),
            "louvain_community": community[i],
        }
        for i in range(N)
    }


# ── TF-IDF — pure Python, no sklearn ─────────────────────────────────────────
def build_tfidf(class_tokens):
    N  = len(class_tokens)
    df = defaultdict(int)
    for tokens in class_tokens.values():
        for tok in set(tokens):
            df[tok] += 1

    tfidf = {}
    for fqn, tokens in class_tokens.items():
        tf  = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        vec = {}
        for tok, count in tf.items():
            idf = math.log((N + 1) / (df[tok] + 1)) + 1
            vec[f"tfidf_{tok}"] = round(count * idf, 4)
        tfidf[fqn] = vec
    return tfidf


# ── Pure-Python z-score normalisation (column-wise) ──────────────────────────
def zscore_normalize(rows, numeric_cols):
    """
    rows: list of dicts.
    Returns list of dicts with numeric_cols z-score normalised.
    No numpy — uses plain math.
    """
    # compute mean and std per column
    sums   = defaultdict(float)
    counts = defaultdict(int)
    for row in rows:
        for col in numeric_cols:
            val = row.get(col, 0)
            if val is not None:
                sums[col]   += val
                counts[col] += 1

    means = {col: sums[col] / (counts[col] or 1) for col in numeric_cols}

    sq_sums = defaultdict(float)
    for row in rows:
        for col in numeric_cols:
            diff = row.get(col, 0) - means[col]
            sq_sums[col] += diff * diff

    stds = {
        col: math.sqrt(sq_sums[col] / (counts[col] or 1))
        for col in numeric_cols
    }

    normalised = []
    for row in rows:
        new_row = dict(row)
        for col in numeric_cols:
            std = stds[col]
            new_row[col] = round(
                (row.get(col, 0) - means[col]) / (std if std > 1e-9 else 1.0),
                6
            )
        normalised.append(new_row)
    return normalised


# ── CSV writer (no pandas needed, but kept for compatibility) ─────────────────
def write_csv(rows, path):
    if not rows:
        log.warning("No rows to write to %s", path)
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info("%s: %d rows x %d cols", Path(path).name, len(rows), len(cols))


class FeatureBuilder:
    def __init__(self, output_dir: str):
        self.out = Path(output_dir)

    def run(self):
        log.info("Building feature matrix (OpenBLAS-free) ...")
        cg = json.loads((self.out / "call_graph.json").read_text(encoding="utf-8"))
        db = json.loads((self.out / "db_features.json").read_text(encoding="utf-8"))
        sm = json.loads((self.out / "stereotype_map.json").read_text(encoding="utf-8"))

        nodes = cg["nodes"]
        edges = cg["edges"]
        fqns  = [n["fqn"] for n in nodes]

        # graph topology (pure Python)
        topology = graph_metrics(nodes, edges)

        # is_hub threshold
        fan_outs = [n["fan_out"] for n in nodes]
        if fan_outs:
            mean_fo = sum(fan_outs) / len(fan_outs)
            var_fo  = sum((x - mean_fo) ** 2 for x in fan_outs) / len(fan_outs)
            std_fo  = math.sqrt(var_fo)
        else:
            mean_fo = std_fo = 0

        # DB lookup
        db_by_fqn = {e["fqn"]: e for e in db["classes"]}

        # table co-access
        table_users = defaultdict(set)
        for entry in db["classes"]:
            for tbl in entry["tables"]:
                table_users[tbl].add(entry["fqn"])

        # TF-IDF tokens
        class_tokens = {}
        for n in nodes:
            class_tokens[n["fqn"]] = tokenize_class(
                n.get("methods", []),
                n.get("javadoc", ""),
                n.get("string_literals", []),
            )
        tfidf_vecs = build_tfidf(class_tokens)

        # assemble rows
        rows = []
        for n in nodes:
            fqn      = n["fqn"]
            db_entry = db_by_fqn.get(fqn, {})
            stereo   = sm.get(fqn, {})
            topo     = topology.get(fqn, {})
            tables   = db_entry.get("tables", [])

            fan_out = n["fan_out"]
            fan_in  = n["fan_in"]
            is_hub  = 1 if fan_out > (mean_fo + std_fo) else 0

            table_oh = {
                f"uses_{t.replace('_','')[:10]}": int(t in tables)
                for t in TABLES
            }
            shares = sum(len(table_users[t]) - 1 for t in tables)

            row = {
                "fqn": fqn,
                # structural
                "fan_out":           fan_out,
                "fan_in":            fan_in,
                "is_hub":            is_hub,
                "betweenness":       topo.get("betweenness", 0),
                "pagerank":          topo.get("pagerank", 0),
                "clustering_coeff":  topo.get("clustering_coeff", 0),
                "louvain_community": topo.get("louvain_community", 0),
                # data
                "tables_count":      db_entry.get("tables_count", 0),
                "fk_count":          db_entry.get("totalFKs", 0),
                "shares_table_count": shares,
                **table_oh,
                # stereotype
                "is_controller": int(stereo.get("is_controller", False)),
                "is_service":    int(stereo.get("is_service", False)),
                "is_repository": int(stereo.get("is_repository", False)),
                "is_entity":     int(stereo.get("is_entity", False)),
                "is_config":     int(stereo.get("is_config", False)),
            }
            row.update(tfidf_vecs.get(fqn, {}))
            rows.append(row)

        # fill missing TF-IDF keys with 0
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        for r in rows:
            for k in all_keys:
                if k not in r:
                    r[k] = 0

        write_csv(rows, self.out / "feature_matrix.csv")

        # z-score normalise (pure Python — no numpy)
        non_numeric = {"fqn", "louvain_community"}
        num_cols    = [c for c in rows[0].keys() if c not in non_numeric]
        norm_rows   = zscore_normalize(rows, num_cols)
        write_csv(norm_rows, self.out / "X_features.csv")
