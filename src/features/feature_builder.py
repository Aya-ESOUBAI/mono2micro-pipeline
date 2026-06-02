"""
src/features/feature_builder.py
────────────────────────────────
Reads call_graph.json + db_features.json + stereotype_map.json
and produces:
  outputs/feature_matrix.csv   (raw, before normalisation)
  outputs/X_features.csv       (z-score normalised, ML-ready)

Feature families (§2.2 of the audit):
  Structural  — fan_in, fan_out, is_hub, betweenness, pagerank, clustering_coeff
  Data-access — one-hot per PetClinic table, tables_count, fk_count, shares_table
  Semantic    — TF-IDF on camelCase-split tokens (methods + javadoc + literals)
  Stereotype  — is_controller, is_service, is_repository, is_entity, is_config
"""

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

TABLES = ["owners", "pets", "visits", "vets", "specialties",
          "vet_specialties", "types", "pet_types"]


# ── camelCase tokeniser ────────────────────────────────────────────────────────

def camel_split(token: str) -> list[str]:
    """'OwnerRepository' → ['Owner', 'Repository'] → lower."""
    parts = re.sub(r'([A-Z][a-z]+)', r' \1', token).split()
    return [p.lower() for p in parts if len(p) > 1]


def tokenize_class(methods: list, javadoc: str, literals: list) -> list[str]:
    tokens = []
    for m in methods:
        tokens.extend(camel_split(m))
    tokens.extend(camel_split(javadoc))
    for s in literals:
        tokens.extend(s.lower().split())
    return tokens


# ── simple graph-topology metrics ─────────────────────────────────────────────

def graph_metrics(nodes: list[dict], edges: list[dict]) -> dict[str, dict]:
    """
    Returns per-FQN dict with:
      betweenness_approx, pagerank, clustering_coeff, louvain_community_id
    Uses pure-Python approximations (no networkx required for small graphs).
    For production: `import networkx as nx` and use nx.betweenness_centrality.
    """
    fqns = [n["fqn"] for n in nodes]
    idx  = {fqn: i for i, fqn in enumerate(fqns)}
    N    = len(fqns)

    # adjacency list (directed)
    out_adj: dict[int, set[int]] = defaultdict(set)
    in_adj:  dict[int, set[int]] = defaultdict(set)
    for e in edges:
        fi, ti = idx.get(e["fromClass"], -1), idx.get(e["toClass"], -1)
        if fi >= 0 and ti >= 0:
            out_adj[fi].add(ti)
            in_adj[ti].add(fi)

    # ── PageRank (power iteration, 30 steps) ──────────────────────────────────
    pr = {i: 1.0 / N for i in range(N)}
    d = 0.85
    for _ in range(30):
        new_pr: dict[int, float] = {}
        for i in range(N):
            rank = (1 - d) / N
            for j in in_adj[i]:
                out_deg = len(out_adj[j]) or 1
                rank += d * pr[j] / out_deg
            new_pr[i] = rank
        pr = new_pr

    # ── Clustering coefficient (local, undirected) ────────────────────────────
    undirected: dict[int, set[int]] = defaultdict(set)
    for i, js in out_adj.items():
        for j in js:
            undirected[i].add(j)
            undirected[j].add(i)

    cc: dict[int, float] = {}
    for i in range(N):
        neighbours = undirected[i]
        k = len(neighbours)
        if k < 2:
            cc[i] = 0.0
        else:
            links = sum(
                1 for u in neighbours for v in neighbours
                if u != v and v in undirected[u]
            )
            cc[i] = links / (k * (k - 1))

    # ── Approx betweenness (sum of fraction of shortest paths through node) ───
    # BFS from every source
    betw = [0.0] * N

    def bfs_sp(src):
        from collections import deque
        dist = [-1] * N
        paths = [0] * N
        dist[src] = 0
        paths[src] = 1
        q = deque([src])
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
        return order, dist, paths

    for s in range(N):
        order, dist, paths = bfs_sp(s)
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

    # ── Louvain communities (simple greedy — for illustration) ────────────────
    community = list(range(N))
    changed = True
    while changed:
        changed = False
        for i in range(N):
            neighbours = list(out_adj[i] | in_adj[i])
            if not neighbours:
                continue
            best_c = community[i]
            best_gain = 0.0
            comm_counts: dict[int, int] = defaultdict(int)
            for j in neighbours:
                comm_counts[community[j]] += 1
            for c, cnt in comm_counts.items():
                gain = cnt - comm_counts.get(community[i], 0)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c != community[i]:
                community[i] = best_c
                changed = True

    return {
        fqns[i]: {
            "pagerank":          round(pr[i], 6),
            "clustering_coeff":  round(cc[i], 4),
            "betweenness":       round(betw[i], 6),
            "louvain_community": community[i],
        }
        for i in range(N)
    }


# ── TF-IDF (manual, no sklearn dependency for text) ───────────────────────────

def build_tfidf(class_tokens: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    N = len(class_tokens)
    df: dict[str, int] = defaultdict(int)
    for tokens in class_tokens.values():
        for tok in set(tokens):
            df[tok] += 1

    vocab = sorted(df.keys())
    tfidf: dict[str, dict[str, float]] = {}
    for fqn, tokens in class_tokens.items():
        tf: dict[str, int] = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        vec = {}
        for tok in vocab:
            if tf[tok] > 0:
                idf = math.log((N + 1) / (df[tok] + 1)) + 1
                vec[f"tfidf_{tok}"] = round(tf[tok] * idf, 4)
        tfidf[fqn] = vec
    return tfidf


class FeatureBuilder:
    def __init__(self, output_dir: str):
        self.out = Path(output_dir)

    def run(self):
        log.info("Building feature matrix …")
        cg  = json.loads((self.out / "call_graph.json").read_text())
        db  = json.loads((self.out / "db_features.json").read_text())
        sm  = json.loads((self.out / "stereotype_map.json").read_text())

        nodes  = cg["nodes"]
        edges  = cg["edges"]
        fqns   = [n["fqn"] for n in nodes]

        # ── graph topology metrics ─────────────────────────────────────────────
        topology = graph_metrics(nodes, edges)

        # ── mean fan_out for is_hub ────────────────────────────────────────────
        fan_outs  = [n["fan_out"] for n in nodes]
        mean_fo   = np.mean(fan_outs) if fan_outs else 0
        std_fo    = np.std(fan_outs)  if fan_outs else 1

        # ── DB lookup ─────────────────────────────────────────────────────────
        db_by_fqn = {entry["fqn"]: entry for entry in db["classes"]}

        # ── table co-access: how many other classes share a table ──────────────
        table_users: dict[str, set[str]] = defaultdict(set)
        for entry in db["classes"]:
            for tbl in entry["tables"]:
                table_users[tbl].add(entry["fqn"])

        # ── TF-IDF tokens ─────────────────────────────────────────────────────
        class_tokens: dict[str, list[str]] = {}
        for n in nodes:
            class_tokens[n["fqn"]] = tokenize_class(
                n.get("methods", []),
                n.get("javadoc", ""),
                n.get("string_literals", []),
            )
        tfidf_vecs = build_tfidf(class_tokens)

        # ── assemble rows ──────────────────────────────────────────────────────
        rows = []
        for n in nodes:
            fqn = n["fqn"]
            db_entry = db_by_fqn.get(fqn, {})
            stereo   = sm.get(fqn, {})
            topo     = topology.get(fqn, {})
            tables   = db_entry.get("tables", [])

            fan_out  = n["fan_out"]
            fan_in   = n["fan_in"]
            is_hub   = 1 if fan_out > (mean_fo + std_fo) else 0

            # table one-hots
            table_oh = {f"uses_{t.replace('_','')[:10]}": int(t in tables)
                        for t in TABLES}

            shares_table_count = sum(
                len(table_users[t]) - 1 for t in tables
            )

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
                # data-access
                "tables_count":      db_entry.get("tables_count", 0),
                "fk_count":          db_entry.get("totalFKs", 0),
                "shares_table_count": shares_table_count,
                **table_oh,
                # stereotype
                "is_controller":     int(stereo.get("is_controller", False)),
                "is_service":        int(stereo.get("is_service", False)),
                "is_repository":     int(stereo.get("is_repository", False)),
                "is_entity":         int(stereo.get("is_entity", False)),
                "is_config":         int(stereo.get("is_config", False)),
            }
            # merge TF-IDF
            row.update(tfidf_vecs.get(fqn, {}))
            rows.append(row)

        raw_df = pd.DataFrame(rows).fillna(0)
        raw_df.to_csv(self.out / "feature_matrix.csv", index=False)
        log.info("feature_matrix.csv: %d rows × %d cols", *raw_df.shape)

        # ── z-score normalise numeric columns (exclude fqn & louvain) ─────────
        non_numeric = {"fqn", "louvain_community"}
        num_cols = [c for c in raw_df.columns if c not in non_numeric]
        scaler = StandardScaler()
        X = scaler.fit_transform(raw_df[num_cols])
        X_df = pd.DataFrame(X, columns=num_cols)
        X_df.insert(0, "fqn", raw_df["fqn"].values)
        X_df["louvain_community"] = raw_df["louvain_community"].values
        X_df.to_csv(self.out / "X_features.csv", index=False)
        log.info("X_features.csv: %d rows × %d cols (normalised)", *X_df.shape)
