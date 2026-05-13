package com.extractor;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * ============================================================
 *  ÉTAPE 2B — Extraction des Features DB depuis schema.sql
 * ============================================================
 *
 *  JavaParser analyse le Java → ce module analyse le SQL.
 *  Il lit le fichier schema.sql de spring-petclinic et extrait :
 *    - Tables + colonnes (nom, type, nullable)
 *    - Clés étrangères explicites (FOREIGN KEY ... REFERENCES)
 *    - FK inférées depuis les noms de colonnes (_id suffix)
 *    - Groupes de domaines métier (pour le DDD)
 *
 *  Output : outputs/db_features.json
 * ============================================================
 */
public class DBFeatureExtractor {

    private final String projectPath;
    private final String outputDir = "./outputs";

    // Chemins possibles pour schema.sql dans spring-petclinic
    private static final String[] SCHEMA_PATHS = {
        "/src/main/resources/db/h2/schema.sql",
        "/src/main/resources/schema.sql",
        "/src/main/resources/db/hsqldb/schema.sql",
        "/src/main/resources/db/mysql/schema.sql"
    };

    // ── Modèles de données ──────────────────────────────────
    static class Column {
        String name;
        String type;
        boolean nullable;
        boolean primaryKey;

        Column(String name, String type, boolean nullable, boolean primaryKey) {
            this.name       = name;
            this.type       = type;
            this.nullable   = nullable;
            this.primaryKey = primaryKey;
        }
    }

    static class ForeignKey {
        String column;
        String fromTable;
        String referencesTable;
        String referencesColumn;
        boolean inferred;

        ForeignKey(String column, String fromTable,
                   String referencesTable, String referencesColumn,
                   boolean inferred) {
            this.column          = column;
            this.fromTable       = fromTable;
            this.referencesTable = referencesTable;
            this.referencesColumn = referencesColumn;
            this.inferred        = inferred;
        }
    }

    static class DBResult {
        Map<String, Object>           summary;
        Map<String, List<Column>>     tables;
        List<ForeignKey>              foreignKeys;
        Map<String, List<String>>     tableGraph;
        Map<String, List<String>>     domainGroups;

        DBResult() {
            summary      = new LinkedHashMap<>();
            tables       = new LinkedHashMap<>();
            foreignKeys  = new ArrayList<>();
            tableGraph   = new LinkedHashMap<>();
            domainGroups = new LinkedHashMap<>();
        }
    }

    // ── Patterns SQL ────────────────────────────────────────
    private static final Pattern CREATE_TABLE = Pattern.compile(
            "CREATE\\s+TABLE\\s+(\\w+)\\s*\\((.+?)\\);",
            Pattern.DOTALL | Pattern.CASE_INSENSITIVE
    );
    private static final Pattern FOREIGN_KEY = Pattern.compile(
            "FOREIGN\\s+KEY\\s*\\((\\w+)\\)\\s+REFERENCES\\s+(\\w+)\\s*\\((\\w+)\\)",
            Pattern.CASE_INSENSITIVE
    );
    private static final Pattern INLINE_FK = Pattern.compile(
            "(\\w+)\\s+\\w+.*?REFERENCES\\s+(\\w+)\\s*\\((\\w+)\\)",
            Pattern.CASE_INSENSITIVE
    );

    public DBFeatureExtractor(String projectPath) {
        this.projectPath = projectPath;
    }

    public void extract() throws IOException {

        // Trouver le fichier schema.sql
        String schemaFile = findSchemaFile();
        if (schemaFile == null) {
            System.out.println("⚠️  schema.sql non trouvé → données simulées");
            extractSimulated();
            return;
        }

        System.out.println("📄 Schema SQL trouvé : " + schemaFile);
        String sql = new String(Files.readAllBytes(Paths.get(schemaFile)));
        System.out.println("   Taille : " + sql.length() + " caractères");

        DBResult result = new DBResult();

        // ── Parser les tables ────────────────────────────────
        parseTables(sql, result);

        // ── Parser les FK explicites ─────────────────────────
        parseForeignKeys(sql, result);

        // ── Inférer les FK depuis les noms de colonnes ───────
        inferForeignKeys(result);

        // ── Construire le graphe des tables ─────────────────
        buildTableGraph(result);

        // ── Détecter les domaines métier (DDD) ───────────────
        detectDomains(result);

        // ── Statistiques ────────────────────────────────────
        result.summary.put("totalTables",   result.tables.size());
        result.summary.put("totalFKs",      result.foreignKeys.size());
        result.summary.put("tableNames",    new ArrayList<>(result.tables.keySet()));

        System.out.println("\n📊 Statistiques :");
        System.out.println("   Tables trouvées : " + result.tables.size());
        System.out.println("   Clés étrangères : " + result.foreignKeys.size());

        System.out.println("\n📋 Tables et colonnes :");
        result.tables.forEach((table, cols) -> {
            List<String> colNames = new ArrayList<>();
            cols.forEach(c -> colNames.add(c.name));
            System.out.printf("   %-20s → %s%n", table, colNames);
        });

        System.out.println("\n🔗 Relations FK :");
        result.foreignKeys.forEach(fk ->
            System.out.printf("   %-15s.%-15s → %s%s%n",
                fk.fromTable, fk.column,
                fk.referencesTable,
                fk.inferred ? " (inférée)" : "")
        );

        System.out.println("\n🗂  Domaines DDD détectés :");
        result.domainGroups.forEach((domain, tables) ->
            System.out.printf("   %-20s → %s%n", domain, tables)
        );

        // ── Sauvegarder ─────────────────────────────────────
        saveJson(result, outputDir + "/db_features.json");
    }

    // ── Parsers ─────────────────────────────────────────────

    private void parseTables(String sql, DBResult result) {
        Matcher m = CREATE_TABLE.matcher(sql);
        while (m.find()) {
            String tableName  = m.group(1).toLowerCase();
            String columnsDef = m.group(2);
            List<Column> columns = parseColumns(columnsDef, tableName);
            result.tables.put(tableName, columns);
        }
    }

    private List<Column> parseColumns(String columnsDef, String tableName) {
        List<Column> columns = new ArrayList<>();
        String[] lines = columnsDef.split(",\\s*\\n|,\\n");

        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty()) continue;

            // Ignorer les contraintes
            if (line.toUpperCase().matches("(PRIMARY|FOREIGN|UNIQUE|INDEX|KEY|CONSTRAINT).*"))
                continue;

            String[] parts = line.split("\\s+");
            if (parts.length < 2) continue;

            String colName  = parts[0];
            String colType  = parts[1].replaceAll("\\(.*\\)", ""); // Retirer (255) etc.
            boolean notNull = line.toUpperCase().contains("NOT NULL");
            boolean isPK    = line.toUpperCase().contains("PRIMARY KEY")
                           || colName.equalsIgnoreCase("id");

            columns.add(new Column(colName, colType, !notNull, isPK));
        }
        return columns;
    }

    private void parseForeignKeys(String sql, DBResult result) {
        // Chercher le contexte de table pour chaque FK
        Matcher tableM = CREATE_TABLE.matcher(sql);
        while (tableM.find()) {
            String tableName  = tableM.group(1).toLowerCase();
            String columnsDef = tableM.group(2);

            Matcher fkM = FOREIGN_KEY.matcher(columnsDef);
            while (fkM.find()) {
                result.foreignKeys.add(new ForeignKey(
                        fkM.group(1), tableName,
                        fkM.group(2).toLowerCase(), fkM.group(3),
                        false
                ));
            }
        }
    }

    private void inferForeignKeys(DBResult result) {
        Set<String> tableNames = result.tables.keySet();

        result.tables.forEach((table, columns) -> {
            for (Column col : columns) {
                String colLower = col.name.toLowerCase();
                if (colLower.endsWith("_id") && !colLower.equals("id")) {
                    String refTable = colLower.substring(0, colLower.length() - 3);

                    // Vérifier que la table référencée existe
                    if (tableNames.contains(refTable)) {
                        // Vérifier qu'elle n'est pas déjà explicite
                        boolean alreadyExists = result.foreignKeys.stream()
                                .anyMatch(fk -> fk.fromTable.equals(table)
                                             && fk.column.equals(col.name));
                        if (!alreadyExists) {
                            result.foreignKeys.add(new ForeignKey(
                                    col.name, table, refTable, "id", true
                            ));
                        }
                    }
                }
            }
        });
    }

    private void buildTableGraph(DBResult result) {
        result.foreignKeys.forEach(fk -> {
            result.tableGraph
                    .computeIfAbsent(fk.fromTable, k -> new ArrayList<>())
                    .add(fk.referencesTable);
        });
    }

    private void detectDomains(DBResult result) {
        Map<String, List<String>> keywords = new LinkedHashMap<>();
        keywords.put("veterinarian", Arrays.asList("vet", "specialty", "specialt"));
        keywords.put("owner_pet",    Arrays.asList("owner", "pet"));
        keywords.put("appointment",  Arrays.asList("visit", "appointment"));
        keywords.put("reference",    Arrays.asList("type", "status", "categor"));

        for (String table : result.tables.keySet()) {
            boolean assigned = false;
            for (Map.Entry<String, List<String>> entry : keywords.entrySet()) {
                for (String kw : entry.getValue()) {
                    if (table.contains(kw)) {
                        result.domainGroups
                                .computeIfAbsent(entry.getKey(), k -> new ArrayList<>())
                                .add(table);
                        assigned = true;
                        break;
                    }
                }
                if (assigned) break;
            }
            if (!assigned) {
                result.domainGroups
                        .computeIfAbsent("other", k -> new ArrayList<>())
                        .add(table);
            }
        }
    }

    // ── Données simulées si schema.sql introuvable ──────────
    private void extractSimulated() throws IOException {
        DBResult result = new DBResult();

        result.tables.put("owners",      Arrays.asList(
                new Column("id","INT",false,true),
                new Column("first_name","VARCHAR",false,false),
                new Column("last_name","VARCHAR",false,false),
                new Column("address","VARCHAR",true,false)
        ));
        result.tables.put("pets", Arrays.asList(
                new Column("id","INT",false,true),
                new Column("name","VARCHAR",false,false),
                new Column("birth_date","DATE",true,false),
                new Column("type_id","INT",false,false),
                new Column("owner_id","INT",false,false)
        ));
        result.tables.put("vets", Arrays.asList(
                new Column("id","INT",false,true),
                new Column("first_name","VARCHAR",false,false),
                new Column("last_name","VARCHAR",false,false)
        ));
        result.tables.put("visits", Arrays.asList(
                new Column("id","INT",false,true),
                new Column("pet_id","INT",false,false),
                new Column("visit_date","DATE",true,false),
                new Column("description","VARCHAR",true,false)
        ));
        result.tables.put("specialties", Arrays.asList(
                new Column("id","INT",false,true),
                new Column("name","VARCHAR",false,false)
        ));
        result.tables.put("types", Arrays.asList(
                new Column("id","INT",false,true),
                new Column("name","VARCHAR",false,false)
        ));

        result.foreignKeys.add(new ForeignKey("owner_id","pets","owners","id",false));
        result.foreignKeys.add(new ForeignKey("type_id", "pets","types","id", false));
        result.foreignKeys.add(new ForeignKey("pet_id", "visits","pets","id",false));

        buildTableGraph(result);
        detectDomains(result);
        result.summary.put("totalTables", result.tables.size());
        result.summary.put("totalFKs",    result.foreignKeys.size());
        result.summary.put("simulated",   true);

        saveJson(result, outputDir + "/db_features.json");
        System.out.println("✅ DB features simulées sauvegardées");
    }

    private String findSchemaFile() {
        for (String path : SCHEMA_PATHS) {
            String full = projectPath + path;
            if (new File(full).exists()) return full;
        }
        return null;
    }

    private void saveJson(Object obj, String path) throws IOException {
        new File(outputDir).mkdirs();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (Writer w = new FileWriter(path)) {
            gson.toJson(obj, w);
        }
        System.out.println("\n✅ DB features sauvegardées → " + path);
    }
}
