# JavaParser Feature Extractor
## Monolith → Microservices | spring-petclinic

---

## Pourquoi JavaParser ?

| | javalang (Python) | JavaParser (Java) |
|---|---|---|
| Langage | Python | **Java natif** |
| Résolution de types | ❌ Non | ✅ Oui (Symbol Solver) |
| Support Java 11+ | Partiel | ✅ Complet |
| Précision AST | Basique | ✅ Complète |
| Annotations Spring | Manuelle | ✅ Native |

---

## Structure du projet

```
javaparser-extractor/
├── pom.xml                                    ← Maven + dépendances
├── run.bat                                    ← Script Windows
└── src/main/java/com/extractor/
    ├── Main.java                              ← Point d'entrée
    ├── CallGraphExtractor.java                ← Étape 2A
    ├── DBFeatureExtractor.java                ← Étape 2B
    ├── LogExtractor.java                      ← Étape 2C
    └── FeatureMatrixBuilder.java              ← Étape 2D
```

---

## Prérequis

- Java 11+ : https://adoptium.net
- Maven 3.6+ : https://maven.apache.org/download.cgi

Vérifier :
```
java -version
mvn -version
```

---

## Installation et lancement

### Option 1 — Script automatique (recommandé)

```bat
cd javaparser-extractor
run.bat C:\Users\hp\Downloads\spring-petclinic
```

### Option 2 — Commandes manuelles

```bat
REM 1. Compiler
mvn clean package

REM 2. Lancer
java -jar target\extractor-all.jar C:\Users\hp\Downloads\spring-petclinic
```

---

## Ce que fait chaque fichier Java

### `Main.java`
Point d'entrée. Lance les 4 étapes dans l'ordre.

### `CallGraphExtractor.java` — Étape 2A
Utilise JavaParser pour :
- Parser chaque `.java` en AST
- Visiter les `MethodCallExpr` → extraire `fromClass → toClass.method()`
- Visiter les `FieldDeclaration` avec `@Autowired` → injections Spring
- Construire la matrice d'adjacence des classes

**Résultat :** `outputs/call_graph.json`
```json
{
  "calls": [
    {"fromClass":"OwnerController","toClass":"ownerRepository","methodName":"findByLastName","type":"method_call"},
    {"fromClass":"OwnerController","toClass":"OwnerRepository", "methodName":"ownerRepository","type":"autowired"}
  ],
  "adjacencyMatrix": {
    "OwnerController": ["OwnerRepository", "PetRepository"]
  }
}
```

### `DBFeatureExtractor.java` — Étape 2B
Lit le fichier `schema.sql` et extrait :
- Tables + colonnes (nom, type, nullable, PK)
- Clés étrangères (FOREIGN KEY ... REFERENCES)
- FK inférées depuis les noms (`owner_id` → table `owners`)
- Groupes de domaines métier (pour la validation DDD)

**Résultat :** `outputs/db_features.json`
```json
{
  "tables": {
    "owners": [{"name":"id","type":"INT","primaryKey":true}, ...],
    "pets":   [{"name":"owner_id","type":"INT","nullable":false}, ...]
  },
  "foreignKeys": [
    {"column":"owner_id","fromTable":"pets","referencesTable":"owners","inferred":false}
  ],
  "domainGroups": {
    "owner_pet":    ["owners","pets","types"],
    "veterinarian": ["vets","specialties"],
    "appointment":  ["visits"]
  }
}
```

### `LogExtractor.java` — Étape 2C
Parse `outputs/petclinic.log` et extrait :
- Requêtes HTTP : méthode + endpoint + ressource
- Requêtes SQL : opération + table
- Corrélation : `GET /owners` → `SELECT owners`

**Résultat :** `outputs/logs_features.json`

### `FeatureMatrixBuilder.java` — Étape 2D
Fusionne les 3 sources en une matrice :

**Résultat :** `outputs/feature_matrix.csv`

```
class_name        | calls_out | calls_in | db_columns | db_fk_out | log_frequency | is_controller | is_repository
OwnerController   |     5     |    0     |     0      |     0     |      8        |       1       |       0
OwnerRepository   |     2     |    3     |     5      |     1     |      8        |       0       |       1
VetController     |     3     |    0     |     0      |     0     |      4        |       1       |       0
VetRepository     |     1     |    2     |     4      |     2     |      4        |       0       |       1
```

---

## Dépendances (pom.xml)

| Bibliothèque | Version | Rôle |
|---|---|---|
| `javaparser-symbol-solver-core` | 3.25.5 | Analyser le code Java (AST) |
| `gson` | 2.10.1 | Écrire les fichiers JSON |
| `commons-csv` | 1.10.0 | Écrire la feature matrix CSV |

---

## Prochaine étape

```
feature_matrix.csv  →  Clustering (K-Means / DBSCAN)  →  Microservices
```
