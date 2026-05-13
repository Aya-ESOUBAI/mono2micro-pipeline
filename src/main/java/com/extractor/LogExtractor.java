package com.extractor;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;
import java.util.stream.Collectors;

/**
 * ============================================================
 *  ÉTAPE 2C — Extraction des Logs d'exécution
 * ============================================================
 *
 *  Parse les logs générés par spring-petclinic et extrait :
 *    - Requêtes HTTP (GET/POST + endpoint + ressource)
 *    - Requêtes SQL  (SELECT/INSERT/UPDATE + table)
 *    - Corrélations HTTP ↔ SQL (qui accède à quelle table)
 *
 *  Si le log est absent → génère des logs simulés réalistes.
 *
 *  Output : outputs/logs_features.json
 * ============================================================
 */
public class LogExtractor {

    private final String logFile   = "./outputs/petclinic.log";
    private final String outputDir = "./outputs";

    // ── Patterns de détection ───────────────────────────────
    private static final Pattern HTTP_PATTERN = Pattern.compile(
            "(GET|POST|PUT|DELETE|PATCH)\\s+(/[^\\s,\"]*)"
    );
    private static final Pattern SQL_PATTERN = Pattern.compile(
            "(select|insert|update|delete)\\s+.*?(from|into|set)\\s+(\\w+)",
            Pattern.CASE_INSENSITIVE
    );

    // ── Modèles de données ──────────────────────────────────
    static class LogEvent {
        String type;       // "HTTP" ou "SQL"
        String method;     // GET, POST, ...
        String endpoint;   // /owners, /vets, ...
        String resource;   // owners, vets, ...
        String table;      // owners, pets, ...
        String operation;  // SELECT, INSERT, ...
        int    lineNumber;

        LogEvent(String type, String method, String endpoint,
                 String resource, String table, String operation,
                 int lineNumber) {
            this.type       = type;
            this.method     = method;
            this.endpoint   = endpoint;
            this.resource   = resource;
            this.table      = table;
            this.operation  = operation;
            this.lineNumber = lineNumber;
        }
    }

    static class LogResult {
        Map<String, Object>        summary;
        List<LogEvent>             events;
        Map<String, Integer>       endpointFrequency;
        Map<String, Integer>       tableFrequency;
        Map<String, List<String>>  resourceTableMap;

        LogResult() {
            summary           = new LinkedHashMap<>();
            events            = new ArrayList<>();
            endpointFrequency = new LinkedHashMap<>();
            tableFrequency    = new LinkedHashMap<>();
            resourceTableMap  = new LinkedHashMap<>();
        }
    }

    public void extract() throws IOException {

        LogResult result = new LogResult();

        // Parser le log ou générer des données simulées
        if (new File(logFile).exists()) {
            System.out.println("📄 Fichier log trouvé : " + logFile);
            parseLogFile(result);
        } else {
            System.out.println("⚠️  Log introuvable → données simulées");
            generateSimulatedLogs(result);
        }

        // ── Calculer les statistiques ────────────────────────
        computeStatistics(result);

        // ── Afficher les résultats ───────────────────────────
        long httpCount = result.events.stream()
                .filter(e -> "HTTP".equals(e.type)).count();
        long sqlCount  = result.events.stream()
                .filter(e -> "SQL".equals(e.type)).count();

        System.out.println("\n📊 Statistiques :");
        System.out.println("   Événements HTTP : " + httpCount);
        System.out.println("   Événements SQL  : " + sqlCount);

        System.out.println("\n🌐 Endpoints les plus appelés :");
        result.endpointFrequency.entrySet().stream()
                .limit(5)
                .forEach(e -> System.out.printf("   %-40s → %dx%n",
                        e.getKey(), e.getValue()));

        System.out.println("\n🗄  Tables les plus accédées :");
        result.tableFrequency.entrySet().stream()
                .limit(5)
                .forEach(e -> System.out.printf("   %-20s → %dx%n",
                        e.getKey(), e.getValue()));

        System.out.println("\n🔗 Corrélation Ressource HTTP → Table SQL :");
        result.resourceTableMap.forEach((res, tables) ->
                System.out.printf("   %-15s → %s%n", res, tables));

        // ── Sauvegarder ─────────────────────────────────────
        saveJson(result, outputDir + "/logs_features.json");
    }

    // ── Parsing du fichier log ───────────────────────────────

    private void parseLogFile(LogResult result) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(logFile));
        System.out.println("   " + lines.size() + " lignes lues");

        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            int lineNum = i + 1;

            // Requêtes HTTP
            Matcher httpM = HTTP_PATTERN.matcher(line);
            if (httpM.find()) {
                String endpoint = httpM.group(2);
                String resource = extractResource(endpoint);
                result.events.add(new LogEvent(
                        "HTTP", httpM.group(1), endpoint,
                        resource, null, null, lineNum
                ));
            }

            // Requêtes SQL
            Matcher sqlM = SQL_PATTERN.matcher(line);
            if (sqlM.find()) {
                result.events.add(new LogEvent(
                        "SQL", null, null, null,
                        sqlM.group(3).toLowerCase(),
                        sqlM.group(1).toUpperCase(),
                        lineNum
                ));
            }
        }
    }

    private String extractResource(String endpoint) {
        String clean = endpoint.startsWith("/") ? endpoint.substring(1) : endpoint;
        return clean.split("/")[0].split("\\?")[0];
    }

    // ── Logs simulés ─────────────────────────────────────────

    private void generateSimulatedLogs(LogResult result) {
        Object[][] simulated = {
            // type,  method, endpoint,                    resource,  table,    operation
            {"HTTP","GET",  "/owners",                    "owners",  null,     null},
            {"SQL", null,   null,                         null,      "owners", "SELECT"},
            {"HTTP","GET",  "/owners/1",                  "owners",  null,     null},
            {"SQL", null,   null,                         null,      "owners", "SELECT"},
            {"SQL", null,   null,                         null,      "pets",   "SELECT"},
            {"HTTP","POST", "/owners/new",                "owners",  null,     null},
            {"SQL", null,   null,                         null,      "owners", "INSERT"},
            {"HTTP","GET",  "/owners/1/pets/new",         "pets",    null,     null},
            {"HTTP","POST", "/owners/1/pets/new",         "pets",    null,     null},
            {"SQL", null,   null,                         null,      "pets",   "INSERT"},
            {"HTTP","GET",  "/owners/1/pets/1/visits/new","visits",  null,     null},
            {"HTTP","POST", "/owners/1/pets/1/visits/new","visits",  null,     null},
            {"SQL", null,   null,                         null,      "visits", "INSERT"},
            {"HTTP","GET",  "/vets",                      "vets",    null,     null},
            {"SQL", null,   null,                         null,      "vets",   "SELECT"},
            {"SQL", null,   null,                         null,      "specialties","SELECT"},
            {"HTTP","GET",  "/owners?lastName=Davis",     "owners",  null,     null},
            {"SQL", null,   null,                         null,      "owners", "SELECT"},
        };

        for (int i = 0; i < simulated.length; i++) {
            Object[] s = simulated[i];
            result.events.add(new LogEvent(
                    (String)s[0], (String)s[1], (String)s[2],
                    (String)s[3], (String)s[4], (String)s[5],
                    i + 1
            ));
        }

        System.out.println("   " + result.events.size() + " événements simulés générés");
    }

    // ── Statistiques ─────────────────────────────────────────

    private void computeStatistics(LogResult result) {
        // Fréquence des endpoints
        result.events.stream()
                .filter(e -> "HTTP".equals(e.type) && e.endpoint != null)
                .forEach(e -> result.endpointFrequency.merge(e.endpoint, 1, Integer::sum));

        // Fréquence des tables
        result.events.stream()
                .filter(e -> "SQL".equals(e.type) && e.table != null)
                .forEach(e -> result.tableFrequency.merge(e.table, 1, Integer::sum));

        // Corrélation ressource → table
        String prevResource = null;
        for (LogEvent event : result.events) {
            if ("HTTP".equals(event.type)) {
                prevResource = event.resource;
            } else if ("SQL".equals(event.type) && prevResource != null && event.table != null) {
                result.resourceTableMap
                        .computeIfAbsent(prevResource, k -> new ArrayList<>())
                        .add(event.table);
            }
        }
        // Dédupliquer
        result.resourceTableMap.replaceAll((k, v) -> new ArrayList<>(new LinkedHashSet<>(v)));

        long httpCount = result.events.stream().filter(e -> "HTTP".equals(e.type)).count();
        long sqlCount  = result.events.stream().filter(e -> "SQL".equals(e.type)).count();
        result.summary.put("totalHTTP", httpCount);
        result.summary.put("totalSQL",  sqlCount);
        result.summary.put("totalEvents", result.events.size());
    }

    private void saveJson(Object obj, String path) throws IOException {
        new File(outputDir).mkdirs();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (Writer w = new FileWriter(path)) {
            gson.toJson(obj, w);
        }
        System.out.println("\n✅ Logs sauvegardés → " + path);
    }
}
