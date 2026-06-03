# Mono2Micro Pipeline — Spring PetClinic

Unsupervised clustering pipeline for **Monolith → Microservices decomposition**,
implementing all recommendations from the PhD audit report.

---

## Architecture

```
mono2micro-pipeline/
├── pipeline.py                          # Main orchestrator (CLI)
├── requirements.txt
├── src/
│   ├── extractor/
│   │   ├── java_extractor.py            # Step 1 — Java parsing (SymbolSolver-ready)
│   │   └── log_parser.py               # Step 2 — Real runtime trace parser
│   ├── features/
│   │   ├── feature_builder.py           # Step 3 — Feature matrix (all non-zero)
│   │   └── similarity_matrix.py        # Step 4 — Hybrid D_similarity
│   ├── clustering/
│   │   └── clusterer.py                # Step 5 — HAC + Louvain + Spectral
│   └── evaluation/
│       └── evaluator.py                # Step 6 — ARI/NMI/SM/ICP/Q report
└── outputs/
    ├── call_graph.json                  (fixed: fan_in, FQNs, all stereotypes)
    ├── db_features.json                 (inferred FKs, domain groups)
    ├── stereotype_map.json
    ├── logs_features.json               (real traces from JMeter/k6)
    ├── feature_matrix.csv               (raw, 30–50 columns, none constant-zero)
    ├── X_features.csv                   (z-score normalised)
    ├── D_similarity.npz                 (N×N hybrid matrix) ← real clustering input
    ├── ground_truth.csv                 (class → target microservice)
    └── clustering_results/
        ├── hac_labels.csv
        ├── louvain_labels.csv
        ├── spectral_labels.csv
        ├── dendrogram.png
        └── evaluation_report.md         (ARI, NMI, SM, ICP, Q, Silhouette …)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (demo — uses synthetic PetClinic data)
```bash
python pipeline.py all
```

### 3. Run on real PetClinic sources
```bash
# Clone PetClinic
git clone https://github.com/spring-projects/spring-petclinic.git

# Run pipeline on real sources
python pipeline.py all \
  --src spring-petclinic/src/main/java \
  --output outputs \
  --n-clusters 5
```

### 4. Step-by-step execution
```bash
python pipeline.py extract    # → call_graph.json, db_features.json, stereotype_map.json
python pipeline.py features   # → feature_matrix.csv, X_features.csv
python pipeline.py similarity # → D_similarity.npz / .csv
python pipeline.py cluster    # → hac/louvain/spectral labels + dendrogram
python pipeline.py evaluate   # → evaluation_report.md
```

### 5. Tune similarity weights
```bash
python pipeline.py all \
  --alpha 0.35 \   # structural
  --beta  0.30 \   # data-access
  --gamma 0.20 \   # semantic
  --delta 0.15     # behavioral
```

---

## Step 2 — Real Runtime Traces (mandatory for behavioral features)

Run PetClinic with debug logging:
```properties
# src/main/resources/application.properties
logging.level.org.springframework.web=DEBUG
logging.level.org.hibernate.SQL=DEBUG
```

Generate load with k6:
```bash
k6 run scripts/petclinic_scenario.js
```

Then parse the log:
```bash
python -c "
from src.extractor.log_parser import LogParser
LogParser('app.log', 'outputs').run()
"
```

Re-run from `similarity` onward:
```bash
python pipeline.py similarity && python pipeline.py cluster && python pipeline.py evaluate
```

---

## Step 3 — CodeBERT Semantic Embeddings (optional, stronger semantic signal)

```python
from sentence_transformers import SentenceTransformer
import pandas as pd, numpy as np

model = SentenceTransformer("microsoft/codebert-base")
df = pd.read_csv("outputs/feature_matrix.csv")
texts = df["fqn"].apply(lambda fqn: fqn.split(".")[-1]).tolist()
embeddings = model.encode(texts, normalize_embeddings=True)

np.save("outputs/codebert_embeddings.npy", embeddings)
```

Then pass as `gamma` component in `SimilarityMatrixBuilder`.

---

## Evaluation Metrics Reference

| Metric | Source | Better when |
|--------|--------|-------------|
| ARI | External (ground truth) | → 1.0 |
| NMI | External | → 1.0 |
| Purity | External | → 1.0 |
| SM (structural modularity) | Call graph | → 1.0 (high cohesion) |
| ICP (inter-call %) | Call graph | → 0.0 (low coupling) |
| Modularity Q | Graph theory | → 1.0 |
| Silhouette | Internal | → 1.0 |
| Davies-Bouldin | Internal | → 0.0 |
| Calinski-Harabasz | Internal | higher |

---

## PFA Audit Checklist

- [x] SymbolSolver-ready extractor; `toClass` resolved to FQNs
- [x] Inverse edges built → `fan_in` non-zero
- [x] All Spring stereotypes detected (`@Service`, `@Repository`, `@Entity` …)
- [x] FK inference implemented (`owner_id` → `owners`)
- [x] Repository interfaces included in parsing
- [x] Real runtime trace parser (`log_parser.py`) — replace placeholder with JMeter output
- [x] TF-IDF expanded to method names + JavaDoc + string literals (camelCase-split)
- [x] Hybrid similarity matrix `D_similarity` with α/β/γ/δ weights
- [x] Ground truth labels from `spring-petclinic-microservices` (built-in + CSV override)
- [x] 3 algorithms: HAC + Louvain + Spectral
- [x] Evaluation: ARI/NMI/Purity + SM/ICP + Silhouette/DB/CH + Modularity Q
- [ ] Real JMeter/k6 traces (run Step 2 manually)
- [ ] CodeBERT embeddings (optional Step 3 enhancement)
- [ ] Weight grid-search tuning (α/β/γ/δ vs ARI)
