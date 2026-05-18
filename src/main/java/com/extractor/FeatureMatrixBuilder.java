package com.extractor;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

/**
 * ============================================================
 *  Step 2D — Feature Matrix Builder  (Human-Readable Output)
 * ============================================================
 *
 *  Produces THREE CSV files designed to be immediately readable
 *  by a supervisor without any technical background:
 *
 *  1. feature_matrix_readable.csv
 *     One row per class. Column names in plain French.
 *     0/1 replaced by Oui/Non. Numbers explained in header.
 *
 *  2. feature_matrix_by_domain.csv
 *     Same data grouped by detected domain (owner, vet, visit…)
 *     so the supervisor sees candidate microservices at a glance.
 *
 *  3. feature_matrix_raw.csv
 *     Original numeric version kept for the clustering algorithm.
 * ============================================================
 */
public class FeatureMatrixBuilder {

    private final String outputDir     = "./outputs";
    private final String callGraphFile = "./outputs/call_graph.json";
    private final String dbFile        = "./outputs/db_features.json";
    private final String logsFile      = "./outputs/logs_features.json";

    // ── Data model ───────────────────────────────────────────
    static class FeatureRow {
        // Identity
        String className;
        String packageDomain;   // owner / vet / visit / system / other
        String stereotype;      // Controller / Repository / Entity / Utility
        String relatedTable;

        // Call graph (numeric — for raw CSV)
        int callsOut;
        int callsIn;
        int depsUnique;
        int hasAutowired;

        // DB (numeric)
        int dbColumns;
        int dbFkOut;
        int dbFkIn;
        int dbRelations;

        // Logs (numeric)
        int logFrequency;

        // Spring type flags (numeric)
        int isController;
        int isService;
        int isRepository;
    }

    // ── Entry point ──────────────────────────────────────────
    public void build() throws IOException {

        System.out.println("📂 Loading feature sources...");

        JsonObject callData  = loadJson(callGraphFile);
        JsonObject dbData    = loadJson(dbFile);
        JsonObject logsData  = loadJson(logsFile);

        if (callData == null) {
            System.out.println("❌ call_graph.json missing — run Step 2A first.");
            return;
        }

        // ── 1. Class list ─────────────────────────────────────
        JsonArray classesArr = callData.getAsJsonObject("summary")
                                       .getAsJsonArray("classes");
        List<String> classes = new ArrayList<>();
        for (JsonElement e : classesArr) classes.add(e.getAsString());
        System.out.println("   ✅ " + classes.size() + " classes found");

        // ── 2. Call graph metrics ────────────────────────────
        Map<String, Integer> callsOut  = new HashMap<>();
        Map<String, Integer> callsIn   = new HashMap<>();
        Map<String, Integer> depsUniq  = new HashMap<>();
        Map<String, Integer> autowired = new HashMap<>();

        JsonArray calls = callData.getAsJsonArray("calls");
        for (JsonElement el : calls) {
            JsonObject c = el.getAsJsonObject();
            String from = c.get("fromClass").getAsString();
            String to   = c.get("toClass").getAsString();
            String type = c.get("type").getAsString();
            callsOut.merge(from, 1, Integer::sum);
            callsIn.merge(to, 1, Integer::sum);
            depsUniq.putIfAbsent(from, 0);
            if ("autowired".equals(type) || "constructor_injection".equals(type))
                autowired.merge(from, 1, Integer::sum);
        }

        // ── 3. DB metrics ────────────────────────────────────
        Map<String, Integer> dbCols  = new HashMap<>();
        Map<String, Integer> dbFkOut = new HashMap<>();
        Map<String, Integer> dbFkIn  = new HashMap<>();
        Map<String, Integer> dbRelat = new HashMap<>();

        if (dbData != null) {
            JsonObject tables = dbData.getAsJsonObject("tables");
            if (tables != null)
                tables.entrySet().forEach(e ->
                    dbCols.put(e.getKey(), e.getValue().getAsJsonArray().size()));

            JsonArray fks = dbData.getAsJsonArray("foreignKeys");
            if (fks != null)
                for (JsonElement el : fks) {
                    JsonObject fk = el.getAsJsonObject();
                    String ft = getStr(fk, "fromTable");
                    String rt = getStr(fk, "referencesTable");
                    if (ft != null) dbFkOut.merge(ft, 1, Integer::sum);
                    if (rt != null) dbFkIn.merge(rt, 1, Integer::sum);
                }

            JsonObject graph = dbData.getAsJsonObject("tableGraph");
            if (graph != null)
                graph.entrySet().forEach(e ->
                    dbRelat.put(e.getKey(), e.getValue().getAsJsonArray().size()));

            System.out.println("   ✅ " + dbCols.size() + " DB tables");
        }

        // ── 4. Log frequency ────────────────────────────────
        Map<String, Integer> logFreq = new HashMap<>();
        if (logsData != null) {
            JsonObject tf = logsData.getAsJsonObject("tableFrequency");
            if (tf != null)
                tf.entrySet().forEach(e ->
                    logFreq.put(e.getKey(), e.getValue().getAsInt()));
        }

        // ── 5. Build rows ────────────────────────────────────
        List<FeatureRow> rows = new ArrayList<>();
        for (String cls : classes) {
            FeatureRow r     = new FeatureRow();
            r.className      = cls;
            r.packageDomain  = detectDomain(cls);
            r.stereotype     = detectStereotype(cls);
            r.relatedTable   = findRelatedTable(cls, dbCols.keySet());

            r.callsOut       = callsOut.getOrDefault(cls, 0);
            r.callsIn        = callsIn.getOrDefault(cls, 0);
            r.depsUnique     = depsUniq.getOrDefault(cls, 0);
            r.hasAutowired   = autowired.getOrDefault(cls, 0) > 0 ? 1 : 0;

            String t = r.relatedTable;
            r.dbColumns      = t != null ? dbCols.getOrDefault(t, 0)  : 0;
            r.dbFkOut        = t != null ? dbFkOut.getOrDefault(t, 0) : 0;
            r.dbFkIn         = t != null ? dbFkIn.getOrDefault(t, 0)  : 0;
            r.dbRelations    = t != null ? dbRelat.getOrDefault(t, 0) : 0;
            r.logFrequency   = t != null ? logFreq.getOrDefault(t, 0) : 0;

            r.isController   = cls.endsWith("Controller") ? 1 : 0;
            r.isService      = cls.endsWith("Service")    ? 1 : 0;
            r.isRepository   = cls.endsWith("Repository") || cls.endsWith("Repo") ? 1 : 0;
            rows.add(r);
        }

        // ── 6. Print summary ─────────────────────────────────
        printSummary(rows);

        // ── 7. Write the three CSV files ─────────────────────
        new File(outputDir).mkdirs();
        writeReadableCsv(rows);
        writeByDomainCsv(rows);
        writeRawCsv(rows);
    }

    // ════════════════════════════════════════════════════════
    //  CSV 1 — READABLE  (for the supervisor)
    // ════════════════════════════════════════════════════════
    /**
     * Columns (in French, self-explanatory):
     *
     *  Classe Java          – simple class name
     *  Domaine métier       – owner / vet / visit / system
     *  Rôle Spring          – Controller / Repository / Entity / Utility
     *  Table SQL liée       – DB table this class manages
     *  Appels sortants      – how many calls this class makes
     *  Appels entrants      – how many other classes call this one
     *  Dependances injectées– Oui/Non (Spring @Autowired)
     *  Colonnes dans la table – number of columns in the related DB table
     *  Relations entre tables – number of FK links to other tables
     *  Frequence dans les logs– how often this table appears in HTTP logs
     */
    private void writeReadableCsv(List<FeatureRow> rows) throws IOException {
        String path = outputDir + "/feature_matrix_readable.csv";

        try (Writer w = new FileWriter(path);
             CSVPrinter p = new CSVPrinter(w, CSVFormat.DEFAULT.withHeader(
                 "Classe Java",
                 "Domaine metier",
                 "Role Spring",
                 "Table SQL liee",
                 "Appels sortants (nb methodes appelees)",
                 "Appels entrants (nb classes qui appellent cette classe)",
                 "Dependances injectees (Autowired/Constructeur)",
                 "Nb colonnes dans la table SQL",
                 "Nb relations entre tables (cles etrangeres)",
                 "Frequence dans les logs HTTP"
             ))) {

            for (FeatureRow r : rows) {
                p.printRecord(
                    r.className,
                    r.packageDomain,
                    r.stereotype,
                    r.relatedTable != null ? r.relatedTable : "aucune",
                    r.callsOut,
                    r.callsIn,
                    r.hasAutowired == 1 ? "Oui" : "Non",
                    r.dbColumns,
                    r.dbRelations,
                    r.logFrequency
                );
            }
        }
        System.out.println("✅ feature_matrix_readable.csv  → " + path);
    }

    // ════════════════════════════════════════════════════════
    //  CSV 2 — BY DOMAIN  (grouped by candidate microservice)
    // ════════════════════════════════════════════════════════
    /**
     * Groups classes by detected domain and shows
     * which classes would belong to the same microservice.
     * The supervisor sees immediately: "owner domain = 5 classes".
     */
    private void writeByDomainCsv(List<FeatureRow> rows) throws IOException {
        String path = outputDir + "/feature_matrix_by_domain.csv";

        // Sort by domain then by stereotype priority
        List<FeatureRow> sorted = new ArrayList<>(rows);
        sorted.sort(Comparator
            .comparing((FeatureRow r) -> r.packageDomain)
            .thenComparing(r -> stereotypePriority(r.stereotype)));

        try (Writer w = new FileWriter(path);
             CSVPrinter p = new CSVPrinter(w, CSVFormat.DEFAULT.withHeader(
                 "Domaine metier (Microservice candidat)",
                 "Classe Java",
                 "Role Spring",
                 "Table SQL geree",
                 "Couplage sortant (appels vers autres classes)",
                 "Couplage entrant (appels recus)",
                 "Injecte des dependances",
                 "Frequence utilisation (logs)"
             ))) {

            String lastDomain = "";
            for (FeatureRow r : sorted) {
                // Insert a blank separator row between domains
                if (!r.packageDomain.equals(lastDomain) && !lastDomain.isEmpty()) {
                    p.println(); // blank row = visual separator in Excel/LibreOffice
                }
                lastDomain = r.packageDomain;

                p.printRecord(
                    r.packageDomain,
                    r.className,
                    r.stereotype,
                    r.relatedTable != null ? r.relatedTable : "—",
                    r.callsOut,
                    r.callsIn,
                    r.hasAutowired == 1 ? "Oui" : "Non",
                    r.logFrequency
                );
            }
        }
        System.out.println("✅ feature_matrix_by_domain.csv → " + path);
    }

    // ════════════════════════════════════════════════════════
    //  CSV 3 — RAW  (numeric, for clustering algorithm)
    // ════════════════════════════════════════════════════════
    private void writeRawCsv(List<FeatureRow> rows) throws IOException {
        String path = outputDir + "/feature_matrix.csv";

        try (Writer w = new FileWriter(path);
             CSVPrinter p = new CSVPrinter(w, CSVFormat.DEFAULT.withHeader(
                 "class_name", "related_table",
                 "calls_out", "calls_in", "deps_unique", "has_autowired",
                 "db_columns", "db_fk_out", "db_fk_in", "db_relations",
                 "log_frequency",
                 "is_controller", "is_service", "is_repository"
             ))) {

            for (FeatureRow r : rows) {
                p.printRecord(
                    r.className,
                    r.relatedTable != null ? r.relatedTable : "none",
                    r.callsOut, r.callsIn, r.depsUnique, r.hasAutowired,
                    r.dbColumns, r.dbFkOut, r.dbFkIn, r.dbRelations,
                    r.logFrequency,
                    r.isController, r.isService, r.isRepository
                );
            }
        }
        System.out.println("✅ feature_matrix.csv (raw)     → " + path);
    }

    // ════════════════════════════════════════════════════════
    //  Console summary
    // ════════════════════════════════════════════════════════
    private void printSummary(List<FeatureRow> rows) {
        System.out.printf("%n╔══════════════════════════════════════════════════════╗%n");
        System.out.printf("║  Feature Matrix — %2d classes extraites               ║%n", rows.size());
        System.out.printf("╠══════════════════════════════════════════════════════╣%n");
        System.out.printf("║  %-20s  %-14s  %-10s  %-5s ║%n",
            "Classe", "Domaine", "Rôle", "Appels");
        System.out.printf("╠══════════════════════════════════════════════════════╣%n");

        // Group by domain for display
        Map<String, List<FeatureRow>> byDomain = new LinkedHashMap<>();
        for (FeatureRow r : rows)
            byDomain.computeIfAbsent(r.packageDomain, k -> new ArrayList<>()).add(r);

        for (Map.Entry<String, List<FeatureRow>> entry : byDomain.entrySet()) {
            System.out.printf("║  DOMAINE : %-41s ║%n", entry.getKey().toUpperCase());
            for (FeatureRow r : entry.getValue()) {
                System.out.printf("║    %-22s %-14s %3d app. sortants  ║%n",
                    r.className, r.stereotype, r.callsOut);
            }
            System.out.printf("║  %-52s ║%n", "");
        }
        System.out.printf("╚══════════════════════════════════════════════════════╝%n%n");
    }

    // ════════════════════════════════════════════════════════
    //  Helpers
    // ════════════════════════════════════════════════════════

    /** Detect the business domain from the class name */
    private String detectDomain(String cls) {
        String lower = cls.toLowerCase();
        if (lower.contains("owner") || lower.contains("pet") || lower.contains("visit"))
            return "owner-pet-visit";
        if (lower.contains("vet") || lower.contains("specialty"))
            return "veterinaire";
        if (lower.contains("cache") || lower.contains("web") || lower.contains("crash")
         || lower.contains("welcome") || lower.contains("runtime") || lower.contains("application"))
            return "systeme";
        return "autre";
    }

    /** Detect the Spring stereotype from the class name */
    private String detectStereotype(String cls) {
        if (cls.endsWith("Controller"))  return "Controleur";
        if (cls.endsWith("Repository") || cls.endsWith("Repo")) return "Repository";
        if (cls.endsWith("Service"))    return "Service";
        if (cls.endsWith("Formatter"))  return "Utilitaire";
        if (cls.endsWith("Validator"))  return "Validateur";
        if (cls.endsWith("Configuration")) return "Configuration";
        if (cls.endsWith("Application") || cls.endsWith("RuntimeHints")) return "Application";
        // Likely a JPA entity (no suffix)
        return "Entite";
    }

    /** Order for display: Controller first, then Repository, then others */
    private int stereotypePriority(String stereotype) {
        switch (stereotype) {
            case "Controleur":
                return 1;
            case "Repository":
                return 2;
            case "Service":
                return 3;
            case "Entite":
                return 4;
            default:
                return 5;
        }
    }

    /** Match a class name to a DB table by removing Spring suffixes */
    private String findRelatedTable(String cls, Set<String> tableNames) {
        String lower = cls.toLowerCase()
            .replace("controller","").replace("service","")
            .replace("repository","").replace("repo","")
            .replace("formatter","").replace("validator","");
        for (String table : tableNames)
            if (lower.contains(table) || table.contains(lower)) return table;
        return null;
    }

    private String getStr(JsonObject obj, String key) {
        JsonElement el = obj.get(key);
        return (el != null && !el.isJsonNull()) ? el.getAsString() : null;
    }

    private JsonObject loadJson(String path) {
        if (!new File(path).exists()) {
            System.out.println("   ⚠️  Missing : " + path);
            return null;
        }
        try {
            String content = new String(Files.readAllBytes(Paths.get(path)));
            return new Gson().fromJson(content, JsonObject.class);
        } catch (Exception e) {
            System.out.println("   ❌ Error reading " + path + " : " + e.getMessage());
            return null;
        }
    }
}
