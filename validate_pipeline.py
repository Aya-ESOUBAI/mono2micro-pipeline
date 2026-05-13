"""
╔══════════════════════════════════════════════════════════════╗
║   VALIDATION — Pipeline Monolith → Microservices            ║
║                                                              ║
║   Étape 1 : JavaParser Output                               ║
║             call_graph.json + db_features.json              ║
║             + logs_features.json                            ║
║                  ↓                                          ║
║   Étape 2 : Feature Extraction                              ║
║             feature_matrix.csv (AST + Call Graph + Names)   ║
║                  ↓                                          ║
║   Étape 3 : Normalization + TF-IDF/Embeddings               ║
║             → composite_features.csv  (INPUT Clustering)    ║
╚══════════════════════════════════════════════════════════════╝

  USAGE :
    python validate_pipeline.py

  PRÉREQUIS :
    pip install scikit-learn pandas numpy scipy

  ENTRÉES ATTENDUES (dossier ./outputs/) :
    call_graph.json       ← produit par CallGraphExtractor.java
    db_features.json      ← produit par DBFeatureExtractor.java
    logs_features.json    ← produit par LogExtractor.java
    feature_matrix.csv    ← produit par FeatureMatrixBuilder.java

  SORTIES PRODUITES :
    feature_normalized.csv    ← features numériques normalisées
    tfidf_matrix.csv          ← vecteurs TF-IDF par classe
    composite_features.csv    ← fusion finale → INPUT Clustering
    validation_report.txt     ← rapport complet
"""

import os, re, json, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Chemins (relatifs au dossier où on lance le script) ────
OUTPUTS_DIR        = "./outputs"

# Étape 1 — JavaParser
F_CALL_GRAPH       = "./outputs/call_graph.json"
F_DB_FEATURES      = "./outputs/db_features.json"
F_LOGS             = "./outputs/logs_features.json"

# Étape 2 — Feature Extraction
F_FEATURE_MATRIX   = "./outputs/feature_matrix.csv"

# Étape 3 — Normalization + TF-IDF (générés par ce script)
F_NORMALIZED       = "./outputs/feature_normalized.csv"
F_TFIDF            = "./outputs/tfidf_matrix.csv"
F_COMPOSITE        = "./outputs/composite_features.csv"

# Étape 3 — rapport
F_REPORT           = "./outputs/validation_report.txt"

# Colonnes exactes produites par FeatureMatrixBuilder.java
REQUIRED_COLUMNS = [
    "class_name", "related_table",
    "calls_out", "calls_in", "deps_unique", "has_autowired",
    "db_columns", "db_fk_out", "db_fk_in", "db_relations",
    "log_frequency",
    "is_controller", "is_service", "is_repository"
]

NUMERIC_COLUMNS = [c for c in REQUIRED_COLUMNS
                   if c not in ("class_name", "related_table")]

# Poids de fusion composite
W_NUMERIC = 0.70   # 70% features numériques (call graph + DB + logs)
W_TFIDF   = 0.30   # 30% similarité sémantique (noms CamelCase)


# ═══════════════════════════════════════════════════════════
#   HELPERS D'AFFICHAGE
# ═══════════════════════════════════════════════════════════

def title(text):
    print(f"\n{'═'*62}")
    print(f"  {text}")
    print(f"{'═'*62}")

def ok(msg):    print(f"  ✅  {msg}")
def warn(msg):  print(f"  ⚠️   {msg}")
def fail(msg):  print(f"  ❌  {msg}")
def info(msg):  print(f"      {msg}")
def sep():      print(f"  {'─'*58}")


# ═══════════════════════════════════════════════════════════
#   ÉTAPE 1  — Validation JavaParser Output
# ═══════════════════════════════════════════════════════════

def validate_step1():
    """
    Vérifie les 3 fichiers JSON produits par les extracteurs Java.
    Retourne un dict avec les résultats et le statut.
    """
    title("ÉTAPE 1 — JavaParser Output")
    issues = []

    # ── call_graph.json ────────────────────────────────────
    print("\n  📄 call_graph.json")
    if not os.path.exists(F_CALL_GRAPH):
        fail("Introuvable — lance d'abord :")
        info("java -jar target\\extractor-all.jar <chemin-spring-petclinic>")
        return None, ["call_graph.json manquant"]

    with open(F_CALL_GRAPH, encoding="utf-8") as f:
        cg = json.load(f)

    calls   = cg.get("calls", [])
    summary = cg.get("summary", {})
    classes = summary.get("classes", [])
    adj     = cg.get("adjacencyMatrix", {})

    ok(f"{len(classes)} classes Java analysées")
    ok(f"{len(calls)} appels extraits dans le Call Graph")
    ok(f"{len(adj)} entrées dans la matrice d'adjacence")

    # Types d'appels (method_call vs autowired)
    from collections import Counter
    types = Counter(c.get("type","?") for c in calls)
    info(f"Types : {dict(types)}")

    # Aperçu des 5 premiers appels
    info("Aperçu :")
    for c in calls[:5]:
        info(f"  {c.get('fromClass','?'):25s} → "
             f"{c.get('toClass','?'):25s}.{c.get('methodName','?')}()"
             f"  [ligne {c.get('lineNumber','?')}]")

    if len(classes) == 0:
        warn("Aucune classe — vérifie le chemin vers spring-petclinic")
        issues.append("0 classes dans call_graph.json")

    # ── db_features.json ───────────────────────────────────
    print("\n  📄 db_features.json")
    if not os.path.exists(F_DB_FEATURES):
        warn("Introuvable (optionnel)")
        issues.append("db_features.json manquant")
        db_info = {}
    else:
        with open(F_DB_FEATURES, encoding="utf-8") as f:
            db = json.load(f)
        tables  = db.get("tables", {})
        fks     = db.get("foreignKeys", []) + db.get("inferred_fks", [])
        domains = db.get("domainGroups", {})
        ok(f"{len(tables)} tables SQL : {list(tables.keys())}")
        ok(f"{len(fks)} clés étrangères (explicites + inférées)")
        ok(f"Domaines DDD : {list(domains.keys())}")
        info("Détail des tables :")
        for tbl, cols in tables.items():
            names = [c.get("name","?") if isinstance(c,dict) else str(c) for c in cols]
            info(f"  {tbl:20s} → {names}")
        db_info = {"tables": len(tables), "fks": len(fks), "domains": list(domains.keys())}

    # ── logs_features.json ─────────────────────────────────
    print("\n  📄 logs_features.json")
    if not os.path.exists(F_LOGS):
        warn("Introuvable — données simulées utilisées dans Étape 2")
        issues.append("logs_features.json manquant (logs simulés)")
        log_info = {}
    else:
        with open(F_LOGS, encoding="utf-8") as f:
            lg = json.load(f)
        summary_lg  = lg.get("summary", {})
        res_map     = lg.get("resourceTableMap", {})
        http_cnt    = summary_lg.get("totalHTTP", 0)
        sql_cnt     = summary_lg.get("totalSQL",  0)
        ok(f"{http_cnt} requêtes HTTP extraites")
        ok(f"{sql_cnt} requêtes SQL extraites")
        ok(f"Corrélations HTTP↔SQL : {len(res_map)} ressources")
        info("Corrélations :")
        for res, tbls in res_map.items():
            info(f"  {res:15s} → {tbls}")
        log_info = {"http": http_cnt, "sql": sql_cnt, "correlations": len(res_map)}

    sep()
    ok("ÉTAPE 1 VALIDÉE ✅") if not issues else warn(f"ÉTAPE 1 — {len(issues)} avertissement(s)")

    return {
        "classes":    len(classes),
        "calls":      len(calls),
        "call_types": dict(types),
        "db":         db_info,
        "logs":       log_info,
    }, issues


# ═══════════════════════════════════════════════════════════
#   ÉTAPE 2  — Validation Feature Extraction
# ═══════════════════════════════════════════════════════════

def validate_step2():
    """
    Vérifie feature_matrix.csv : colonnes, valeurs, cohérence.
    Retourne le DataFrame si valide.
    """
    title("ÉTAPE 2 — Feature Extraction (feature_matrix.csv)")
    issues = []

    if not os.path.exists(F_FEATURE_MATRIX):
        fail("feature_matrix.csv introuvable — lance d'abord :")
        info("java -jar target\\extractor-all.jar <chemin-spring-petclinic>")
        return None, ["feature_matrix.csv manquant"]

    df = pd.read_csv(F_FEATURE_MATRIX)
    size_kb = os.path.getsize(F_FEATURE_MATRIX) / 1024
    ok(f"Fichier trouvé ({size_kb:.1f} Ko)")
    ok(f"{len(df)} classes × {len(df.columns)} colonnes")

    # ── Colonnes requises ───────────────────────────────────
    print("\n  📋 Colonnes :")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra   = [c for c in df.columns if c not in REQUIRED_COLUMNS]

    if missing:
        fail(f"Colonnes manquantes : {missing}")
        issues.append(f"Colonnes manquantes : {missing}")
    else:
        ok("Toutes les 14 colonnes requises sont présentes")
    if extra:
        info(f"Colonnes supplémentaires : {extra}")

    # ── Statistiques ────────────────────────────────────────
    print("\n  📊 Statistiques des features numériques :")
    avail_num = [c for c in NUMERIC_COLUMNS if c in df.columns]
    for col in avail_num:
        info(f"  {col:20s} → mean={df[col].mean():.2f}  "
             f"max={int(df[col].max())}  zeros={int((df[col]==0).sum())}/{len(df)}")

    # ── Cohérence ───────────────────────────────────────────
    print("\n  🔍 Vérifications :")

    # Doublons
    dupes = df[df.duplicated(subset=["class_name"])]
    if len(dupes):
        warn(f"{len(dupes)} doublons : {list(dupes['class_name'])}")
        issues.append("Doublons dans class_name")
    else:
        ok("Aucun doublon de classe")

    # Valeurs négatives
    neg = [(c, int(df[c].min())) for c in avail_num if df[c].min() < 0]
    if neg:
        fail(f"Valeurs négatives : {neg}")
        issues.append(f"Valeurs négatives : {neg}")
    else:
        ok("Aucune valeur négative")

    # Classes sans aucun appel
    no_call = df[(df.get("calls_out",pd.Series([1]))==0) &
                 (df.get("calls_in", pd.Series([1]))==0)]
    if len(no_call):
        warn(f"{len(no_call)} classe(s) sans appels : {list(no_call['class_name'])}")
    else:
        ok("Toutes les classes ont au moins un appel")

    # ── Affichage complet ───────────────────────────────────
    print("\n  📋 Feature Matrix complète :")
    disp_cols = ["class_name","related_table",
                 "calls_out","calls_in","db_columns","log_frequency",
                 "is_controller","is_service","is_repository"]
    disp_cols = [c for c in disp_cols if c in df.columns]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df[disp_cols].to_string(index=False))

    sep()
    if not issues:
        ok("ÉTAPE 2 VALIDÉE ✅")
    else:
        warn(f"ÉTAPE 2 — {len(issues)} avertissement(s)")

    return df, issues


# ═══════════════════════════════════════════════════════════
#   ÉTAPE 3  — Normalization + TF-IDF + Composite
# ═══════════════════════════════════════════════════════════

def split_camel(name: str) -> str:
    """
    Découpe un nom CamelCase en tokens de domaine.
    Supprime les suffixes Spring banals.
    Ex : OwnerController → 'owner'
         PetRepository   → 'pet'
         VisitService    → 'visit'
    """
    stop = {"Controller","Service","Repository","Repo",
            "Impl","Manager","Handler","Bean","Helper"}
    parts = re.sub(r"([A-Z][a-z]+)", r" \1", name).split()
    domain = [p.lower() for p in parts if p not in stop]
    return " ".join(domain) if domain else name.lower()


def validate_step3(df: pd.DataFrame):
    """
    Lance la normalisation et le TF-IDF, valide les résultats
    et produit composite_features.csv.
    """
    title("ÉTAPE 3 — Normalization + TF-IDF/Embeddings")
    issues = []

    classes   = df["class_name"].tolist()
    n         = len(classes)
    avail_num = [c for c in NUMERIC_COLUMNS if c in df.columns]
    X_raw     = df[avail_num].fillna(0).astype(float).values

    # ── 3A : StandardScaler ────────────────────────────────
    print("\n  ── 3A : StandardScaler (z-score) sur features numériques")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    df_norm  = pd.DataFrame(X_scaled,
                            columns=[f"num_{c}" for c in avail_num])

    ok(f"{len(avail_num)} features normalisées : {avail_num}")

    # Vérification : moyenne ≈ 0 et std ≈ 1
    means_ok = all(abs(X_scaled.mean(axis=0)) < 0.01)
    stds_after = X_scaled.std(axis=0)
    stds_ok  = bool(((abs(stds_after - 1) < 0.05) | (stds_after < 0.01)).all())

    if means_ok:
        ok("Toutes les moyennes ≈ 0 après normalisation ✅")
    else:
        warn("Certaines moyennes ne sont pas ≈ 0")
        issues.append("Normalisation imparfaite (moyenne)")

    # Features constantes (std = 0 avant normalisation)
    const_feat = [avail_num[i] for i, s in enumerate(scaler.scale_) if s < 0.001]
    if const_feat:
        warn(f"Features constantes (ignorées par scaler) : {const_feat}")
        issues.append(f"Features constantes : {const_feat}")
    else:
        ok("Aucune feature constante ✅")

    info("Avant → Après normalisation :")
    for col, mean, std in zip(avail_num, scaler.mean_, scaler.scale_):
        info(f"  {col:20s}  mean={mean:.2f}  std={std:.2f}  →  mean≈0  std≈1")

    # ── 3B : TF-IDF sur noms de classes ───────────────────
    print("\n  ── 3B : TF-IDF sur noms CamelCase")

    tokenized = [split_camel(c) for c in classes]

    info("Tokenisation :")
    for name, tok in zip(classes[:min(8, n)], tokenized[:min(8, n)]):
        info(f"  {name:30s} → '{tok}'")

    vectorizer = TfidfVectorizer(min_df=1, analyzer="word", ngram_range=(1, 1))
    X_tfidf    = vectorizer.fit_transform(tokenized).toarray()
    vocab      = list(vectorizer.vocabulary_.keys())
    df_tfidf   = pd.DataFrame(X_tfidf,
                              columns=[f"tfidf_{v}" for v in
                                       vectorizer.get_feature_names_out()])

    ok(f"Vocabulaire : {len(vocab)} tokens → {vocab}")
    ok(f"Matrice TF-IDF : {n} × {len(vocab)}")

    # Classes sans aucun token
    zero_rows = [classes[i] for i, row in enumerate(X_tfidf) if row.sum() == 0]
    if zero_rows:
        warn(f"Classes sans token TF-IDF : {zero_rows}")
        issues.append(f"Vecteur TF-IDF nul pour : {zero_rows}")
    else:
        ok("Toutes les classes ont un vecteur TF-IDF non nul ✅")

    # Paires les plus proches sémantiquement
    from scipy.spatial.distance import cdist
    D_cos = cdist(X_tfidf, X_tfidf, metric="cosine")
    np.fill_diagonal(D_cos, 1.0)
    pairs = sorted([
        (D_cos[i][j], classes[i], classes[j])
        for i in range(n) for j in range(i+1, n)
    ])
    info("Classes les plus proches (cosine TF-IDF) :")
    for d, a, b in pairs[:5]:
        info(f"  {a:28s} ↔ {b:28s}  sim={1-d:.3f}")

    # ── 3C : Fusion composite ──────────────────────────────
    print(f"\n  ── 3C : Fusion composite "
          f"(α={W_NUMERIC} × numérique + β={W_TFIDF} × TF-IDF)")

    df_composite = pd.concat([
        df[["class_name", "related_table"]].reset_index(drop=True),
        df_norm   * W_NUMERIC,
        df_tfidf  * W_TFIDF,
    ], axis=1)

    total_feat = len(df_composite.columns) - 2
    ok(f"Composite : {n} classes × {total_feat} features")
    info(f"  {len(avail_num)} features numériques × {W_NUMERIC}")
    info(f"  {len(vocab)} features TF-IDF × {W_TFIDF}")

    # ── Sauvegarde ─────────────────────────────────────────
    print("\n  ── Sauvegarde des fichiers")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    pd.concat([df[["class_name"]], df_norm], axis=1)\
      .to_csv(F_NORMALIZED, index=False)
    ok(f"feature_normalized.csv  → {F_NORMALIZED}")

    pd.concat([df[["class_name"]], df_tfidf], axis=1)\
      .to_csv(F_TFIDF, index=False)
    ok(f"tfidf_matrix.csv        → {F_TFIDF}")

    df_composite.to_csv(F_COMPOSITE, index=False)
    ok(f"composite_features.csv  → {F_COMPOSITE}  ← INPUT Clustering")

    sep()
    if not issues:
        ok("ÉTAPE 3 VALIDÉE ✅")
    else:
        warn(f"ÉTAPE 3 — {len(issues)} avertissement(s)")

    return {
        "n_classes":      n,
        "numeric_feats":  len(avail_num),
        "tfidf_vocab":    len(vocab),
        "total_features": total_feat,
    }, issues


# ═══════════════════════════════════════════════════════════
#   RAPPORT FINAL
# ═══════════════════════════════════════════════════════════

def write_report(r1, i1, r2_meta, i2, r3, i3):
    icon = lambda issues: "✅ OK" if not issues else "⚠️  AVERT."

    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║       RAPPORT DE VALIDATION — PIPELINE              ║",
        "╚══════════════════════════════════════════════════════╝",
        "",
        "  RÉSUMÉ",
        "  " + "─"*52,
        f"  Étape 1 — JavaParser Output       : {icon(i1)}",
        f"  Étape 2 — Feature Extraction      : {icon(i2)}",
        f"  Étape 3 — Normalization + TF-IDF  : {icon(i3)}",
        "",
        "  ÉTAPE 1 — JavaParser",
        "  " + "─"*52,
        f"  Classes analysées   : {r1.get('classes', 0)}",
        f"  Appels extraits     : {r1.get('calls', 0)}",
        f"  Types d'appels      : {r1.get('call_types', {})}",
        f"  Tables SQL          : {r1.get('db', {}).get('tables', 0)}",
        f"  Clés étrangères     : {r1.get('db', {}).get('fks', 0)}",
        f"  Domaines DDD        : {r1.get('db', {}).get('domains', [])}",
        f"  HTTP logs           : {r1.get('logs', {}).get('http', 0)}",
        f"  SQL logs            : {r1.get('logs', {}).get('sql', 0)}",
    ]
    if i1: lines.append(f"  ⚠️  Problèmes : {i1}")

    lines += [
        "",
        "  ÉTAPE 2 — Feature Extraction",
        "  " + "─"*52,
        f"  Classes dans le CSV  : {r2_meta.get('n_classes', 0)}",
        f"  Features numériques  : {NUMERIC_COLUMNS}",
    ]
    if i2: lines.append(f"  ⚠️  Problèmes : {i2}")

    lines += [
        "",
        "  ÉTAPE 3 — Normalization + TF-IDF",
        "  " + "─"*52,
        f"  Classes traitées        : {r3.get('n_classes', 0)}",
        f"  Features numériques     : {r3.get('numeric_feats', 0)}",
        f"  Tokens TF-IDF           : {r3.get('tfidf_vocab', 0)}",
        f"  Features composite total: {r3.get('total_features', 0)}",
        f"  Poids numérique         : {W_NUMERIC}",
        f"  Poids TF-IDF            : {W_TFIDF}",
    ]
    if i3: lines.append(f"  ⚠️  Problèmes : {i3}")

    lines += [
        "",
        "  FICHIERS PRODUITS",
        "  " + "─"*52,
    ]
    all_files = [
        (F_CALL_GRAPH,    "call_graph.json         (Étape 1)"),
        (F_DB_FEATURES,   "db_features.json        (Étape 1)"),
        (F_LOGS,          "logs_features.json      (Étape 1)"),
        (F_FEATURE_MATRIX,"feature_matrix.csv      (Étape 2)"),
        (F_NORMALIZED,    "feature_normalized.csv  (Étape 3)"),
        (F_TFIDF,         "tfidf_matrix.csv        (Étape 3)"),
        (F_COMPOSITE,     "composite_features.csv  (Étape 3) ← INPUT Clustering"),
    ]
    for path, label in all_files:
        exists = os.path.exists(path)
        size   = f"{os.path.getsize(path)/1024:.1f} Ko" if exists else "—"
        icon   = "✅" if exists else "❌"
        lines.append(f"  {icon}  {label:50s} {size}")

    lines += [
        "",
        "  PROCHAINE ÉTAPE",
        "  " + "─"*52,
        "  composite_features.csv est prêt pour le Clustering HAC",
        "  → python step3b_distance_matrix.py",
        "  → python step3c_hac_clustering.py",
    ]

    text = "\n".join(lines)
    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ═══════════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║   VALIDATION DU PIPELINE — 3 ÉTAPES                        ║
║   JavaParser → Feature Extraction → Normalization + TF-IDF ║
╚══════════════════════════════════════════════════════════════╝
""")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Étape 1 ─────────────────────────────────────────────
    r1, i1 = validate_step1()
    if r1 is None:
        print("\n❌ Impossible de continuer sans les fichiers JavaParser.")
        sys.exit(1)

    # ── Étape 2 ─────────────────────────────────────────────
    df, i2 = validate_step2()
    if df is None:
        print("\n❌ Impossible de continuer sans feature_matrix.csv.")
        sys.exit(1)
    r2_meta = {"n_classes": len(df)}

    # ── Étape 3 ─────────────────────────────────────────────
    r3, i3 = validate_step3(df)

    # ── Rapport ─────────────────────────────────────────────
    title("RAPPORT FINAL")
    report = write_report(r1, i1, r2_meta, i2, r3, i3)
    print(report)
    print(f"\n  ✅ Rapport sauvegardé → {F_REPORT}")

    all_good = not (i1 or i2 or i3)
    print("""
╔══════════════════════════════════════════════════════════╗
║  ✅  LES 3 ÉTAPES SONT VALIDÉES                          ║
║                                                          ║
║  composite_features.csv est prêt pour le Clustering !   ║
╚══════════════════════════════════════════════════════════╝
""" if all_good else """
╔══════════════════════════════════════════════════════════╗
║  ⚠️   VALIDATION AVEC AVERTISSEMENTS                     ║
║  → Consulte validation_report.txt pour les détails      ║
╚══════════════════════════════════════════════════════════╝
""")
