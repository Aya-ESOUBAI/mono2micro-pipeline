package com.extractor;

import com.google.gson.*;
import org.apache.commons.csv.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * ============================================================
 *  ÉTAPE 2D — Construction de la Feature Matrix finale
 * ============================================================
 *
 *  Fusionne les 3 sources :
 *    call_graph.json   → features appels inter-classes
 *    db_features.json  → features base de données
 *    logs_features.json→ features logs d'exécution
 *
 *  Produit une ligne par classe Java avec toutes ses features.
 *  C'est le fichier d'INPUT pour le Clustering.
 *
 *  Output : outputs/feature_matrix.csv
 * ============================================================
 */
public class FeatureMatrixBuilder {

    private final String outputDir     = "./outputs";
    private final String callGraphFile = "./outputs/call_graph.json";
    private final String dbFile        = "./outputs/db_features.json";
    private final String logsFile      = "./outputs/logs_features.json";

    // ── Modèle d'une ligne de la feature matrix ─────────────
    static class FeatureRow {
        String  className;
        String  relatedTable;
        // Features Call Graph
        int     callsOut;
        int     callsIn;
        int     depsUnique;
        int     hasAutowired;
        // Features DB
        int     dbColumns;
        int     dbFkOut;
        int     dbFkIn;
        int     dbRelations;
        // Features Logs
        int     logFrequency;
        // Features Spring
        int     isController;
        int     isService;
        int     isRepository;
    }

    public void build() throws IOException {

        // ── Charger les 3 sources ────────────────────────────
        System.out.println("📂 Chargement des features...");

        Gson gson = new Gson();

        // Call graph
        JsonObject callData   = loadJson(callGraphFile);
        JsonObject dbData     = loadJson(dbFile);
        JsonObject logsData   = loadJson(logsFile);

        if (callData == null) {
            System.out.println("❌ call_graph.json manquant → lance step2a d'abord");
            return;
        }

        // ── Extraire les données ──────────────────────────────

        // Classes depuis le call graph
        JsonArray classesArr = callData.getAsJsonObject("summary")
                .getAsJsonArray("classes");
        List<String> classes = new ArrayList<>();
        for (JsonElement e : classesArr) classes.add(e.getAsString());
        System.out.println("   ✅ Call features    : " + classes.size() + " classes");

        // Appels (out et in) par classe
        Map<String, Integer> callsOut = new HashMap<>();
        Map<String, Integer> callsIn  = new HashMap<>();
        Map<String, Integer> depsUniq = new HashMap<>();
        Map<String, Integer> autowired= new HashMap<>();

        JsonArray calls = callData.getAsJsonArray("calls");
        for (JsonElement el : calls) {
            JsonObject call = el.getAsJsonObject();
            String from = call.get("fromClass").getAsString();
            String to   = call.get("toClass").getAsString();
            String type = call.get("type").getAsString();

            callsOut.merge(from, 1, Integer::sum);
            callsIn.merge(to, 1, Integer::sum);
            depsUniq.computeIfAbsent(from, k -> 0);

            if ("autowired".equals(type)) autowired.merge(from, 1, Integer::sum);
        }

        // Tables DB
        Map<String, Integer> dbCols    = new HashMap<>();
        Map<String, Integer> dbFkOut   = new HashMap<>();
        Map<String, Integer> dbFkIn    = new HashMap<>();
        Map<String, Integer> dbRelat   = new HashMap<>();

        if (dbData != null) {
            JsonObject tables = dbData.getAsJsonObject("tables");
            if (tables != null) {
                tables.entrySet().forEach(entry -> {
                    int cols = entry.getValue().getAsJsonArray().size();
                    dbCols.put(entry.getKey(), cols);
                });
            }

            JsonArray fks = dbData.getAsJsonArray("foreignKeys");
            if (fks != null) {
                for (JsonElement el : fks) {
                    JsonObject fk = el.getAsJsonObject();
                    String fromT  = getStr(fk, "fromTable");
                    String refT   = getStr(fk, "referencesTable");
                    if (fromT != null) dbFkOut.merge(fromT, 1, Integer::sum);
                    if (refT  != null) dbFkIn.merge(refT, 1, Integer::sum);
                }
            }

            JsonObject graph = dbData.getAsJsonObject("tableGraph");
            if (graph != null) {
                graph.entrySet().forEach(e ->
                        dbRelat.put(e.getKey(), e.getValue().getAsJsonArray().size())
                );
            }

            System.out.println("   ✅ DB features      : " + dbCols.size() + " tables");
        }

        // Fréquence des tables dans les logs
        Map<String, Integer> logFreq = new HashMap<>();
        if (logsData != null) {
            JsonObject tableFreq = logsData.getAsJsonObject("tableFrequency");
            if (tableFreq != null) {
                tableFreq.entrySet().forEach(e ->
                        logFreq.put(e.getKey(), e.getValue().getAsInt())
                );
            }
            System.out.println("   ✅ Log features     : chargées");
        }

        // ── Construire les lignes de la matrice ───────────────
        List<FeatureRow> rows = new ArrayList<>();

        for (String cls : classes) {
            FeatureRow row = new FeatureRow();
            row.className = cls;

            // Détecter la table liée (heuristique sur le nom de la classe)
            row.relatedTable = findRelatedTable(cls, dbCols.keySet());

            // Features Call Graph
            row.callsOut     = callsOut.getOrDefault(cls, 0);
            row.callsIn      = callsIn.getOrDefault(cls, 0);
            row.depsUnique   = depsUniq.getOrDefault(cls, 0);
            row.hasAutowired = autowired.getOrDefault(cls, 0) > 0 ? 1 : 0;

            // Features DB (via la table liée)
            String t = row.relatedTable;
            row.dbColumns  = t != null ? dbCols.getOrDefault(t, 0)  : 0;
            row.dbFkOut    = t != null ? dbFkOut.getOrDefault(t, 0)  : 0;
            row.dbFkIn     = t != null ? dbFkIn.getOrDefault(t, 0)   : 0;
            row.dbRelations= t != null ? dbRelat.getOrDefault(t, 0)  : 0;

            // Features Logs
            row.logFrequency = t != null ? logFreq.getOrDefault(t, 0) : 0;

            // Features Spring (détection par nom de classe)
            row.isController  = cls.endsWith("Controller") ? 1 : 0;
            row.isService     = cls.endsWith("Service")    ? 1 : 0;
            row.isRepository  = cls.endsWith("Repository") || cls.endsWith("Repo") ? 1 : 0;

            rows.add(row);
        }

        // ── Afficher la matrice ───────────────────────────────
        System.out.printf("\n📊 Feature Matrix : %d classes × 13 features%n", rows.size());
        System.out.printf("%n%-25s %-15s %5s %5s %5s %5s %5s %5s %5s%n",
                "class_name", "table", "c_out","c_in","db_col","fk_out","log","ctrl","repo");
        System.out.println("-".repeat(90));

        rows.forEach(r -> System.out.printf(
                "%-25s %-15s %5d %5d %5d %5d %5d %5d %5d%n",
                r.className,
                r.relatedTable != null ? r.relatedTable : "none",
                r.callsOut, r.callsIn,
                r.dbColumns, r.dbFkOut,
                r.logFrequency,
                r.isController, r.isRepository
        ));

        // ── Sauvegarder en CSV ────────────────────────────────
        saveCsv(rows);
    }

    // ── Utilitaires ─────────────────────────────────────────

    private String findRelatedTable(String className, Set<String> tableNames) {
        String lower = className.toLowerCase()
                .replace("controller", "")
                .replace("service", "")
                .replace("repository", "")
                .replace("repo", "");

        for (String table : tableNames) {
            if (lower.contains(table) || table.contains(lower)) {
                return table;
            }
        }
        return null;
    }

    private String getStr(JsonObject obj, String key) {
        JsonElement el = obj.get(key);
        return (el != null && !el.isJsonNull()) ? el.getAsString() : null;
    }

    private JsonObject loadJson(String path) {
        if (!new File(path).exists()) {
            System.out.println("   ⚠️  Manquant : " + path);
            return null;
        }
        try {
            String content = new String(Files.readAllBytes(Paths.get(path)));
            return new Gson().fromJson(content, JsonObject.class);
        } catch (Exception e) {
            System.out.println("   ❌ Erreur lecture " + path + " : " + e.getMessage());
            return null;
        }
    }

    private void saveCsv(List<FeatureRow> rows) throws IOException {
        String csvPath = outputDir + "/feature_matrix.csv";
        new File(outputDir).mkdirs();

        try (Writer writer = new FileWriter(csvPath);
             CSVPrinter printer = new CSVPrinter(writer,
                     CSVFormat.DEFAULT.withHeader(
                             "class_name", "related_table",
                             "calls_out", "calls_in", "deps_unique", "has_autowired",
                             "db_columns", "db_fk_out", "db_fk_in", "db_relations",
                             "log_frequency",
                             "is_controller", "is_service", "is_repository"
                     ))) {

            for (FeatureRow r : rows) {
                printer.printRecord(
                        r.className,
                        r.relatedTable != null ? r.relatedTable : "none",
                        r.callsOut, r.callsIn, r.depsUnique, r.hasAutowired,
                        r.dbColumns, r.dbFkOut, r.dbFkIn, r.dbRelations,
                        r.logFrequency,
                        r.isController, r.isService, r.isRepository
                );
            }
        }

        System.out.println("\n✅ Feature Matrix sauvegardée → " + csvPath);
        System.out.println("\n🎯 Prochaine étape : Clustering (K-Means/DBSCAN) sur feature_matrix.csv");
    }
}
