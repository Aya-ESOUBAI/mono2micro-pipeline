package com.extractor;

/**
 * ============================================================
 *  Point d'entrée principal — Lance toutes les étapes
 * ============================================================
 *  Usage :
 *    java -jar extractor-all.jar <chemin-vers-spring-petclinic>
 *
 *  Exemple :
 *    java -jar extractor-all.jar C:/Users/hp/Downloads/spring-petclinic
 * ============================================================
 */
public class Main {

    public static void main(String[] args) throws Exception {

        String projectPath = args.length > 0
                ? args[0]
                : "./spring-petclinic";   // chemin par défaut

        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║   MONOLITH → FEATURE EXTRACTION (JavaParser)        ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");
        System.out.println("📂 Projet analysé : " + projectPath);
        System.out.println();

        // ── ÉTAPE 2A : Call Graph ──────────────────────────────
        System.out.println("══════════════════════════════════════════");
        System.out.println("  ÉTAPE 2A — Call Graph (appels entre classes)");
        System.out.println("══════════════════════════════════════════");
        CallGraphExtractor callExtractor = new CallGraphExtractor(projectPath);
        callExtractor.extract();
        System.out.println();

        // ── ÉTAPE 2B : DB Features ────────────────────────────
        System.out.println("══════════════════════════════════════════");
        System.out.println("  ÉTAPE 2B — DB Features (schéma SQL)");
        System.out.println("══════════════════════════════════════════");
        DBFeatureExtractor dbExtractor = new DBFeatureExtractor(projectPath);
        dbExtractor.extract();
        System.out.println();

        // ── ÉTAPE 2C : Logs ───────────────────────────────────
        System.out.println("══════════════════════════════════════════");
        System.out.println("  ÉTAPE 2C — Logs d'exécution");
        System.out.println("══════════════════════════════════════════");
        LogExtractor logExtractor = new LogExtractor();
        logExtractor.extract();
        System.out.println();

        // ── ÉTAPE 2D : Feature Matrix ─────────────────────────
        System.out.println("══════════════════════════════════════════");
        System.out.println("  ÉTAPE 2D — Feature Matrix finale");
        System.out.println("══════════════════════════════════════════");
        FeatureMatrixBuilder matrixBuilder = new FeatureMatrixBuilder();
        matrixBuilder.build();
        System.out.println();

        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║  ✅ Extraction terminée !                            ║");
        System.out.println("║                                                      ║");
        System.out.println("║  Fichiers produits dans ./outputs/ :                 ║");
        System.out.println("║    call_graph.json                                   ║");
        System.out.println("║    db_features.json                                  ║");
        System.out.println("║    logs_features.json                                ║");
        System.out.println("║    feature_matrix.csv  ← INPUT pour Clustering       ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");
    }
}
