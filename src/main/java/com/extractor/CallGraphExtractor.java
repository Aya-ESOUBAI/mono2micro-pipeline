package com.extractor;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * ============================================================
 *  ÉTAPE 2A — Extraction du Call Graph avec JavaParser
 * ============================================================
 *
 *  Ce que fait JavaParser ici :
 *  1. Parse chaque fichier .java en AST (Abstract Syntax Tree)
 *  2. Visite les noeuds de type MethodCallExpr (appels de méthodes)
 *  3. Extrait : classe source, classe cible, nom de méthode
 *  4. Détecte les @Autowired (injections de dépendances Spring)
 *
 *  Output : outputs/call_graph.json
 * ============================================================
 */
public class CallGraphExtractor {

    private final String projectPath;
    private final String srcPath;
    private final String outputDir = "./outputs";

    // Modèle de données pour un appel
    static class MethodCall {
        String fromClass;
        String fromPackage;
        String toClass;
        String methodName;
        String type;  // "method_call" ou "autowired"
        int lineNumber;

        MethodCall(String fromClass, String fromPackage,
                   String toClass, String methodName,
                   String type, int lineNumber) {
            this.fromClass   = fromClass;
            this.fromPackage = fromPackage;
            this.toClass     = toClass;
            this.methodName  = methodName;
            this.type        = type;
            this.lineNumber  = lineNumber;
        }
    }

    // Résultat global
    static class CallGraphResult {
        Map<String, Object> summary;
        List<MethodCall>    calls;
        Map<String, Set<String>> adjacencyMatrix;

        CallGraphResult() {
            summary         = new LinkedHashMap<>();
            calls           = new ArrayList<>();
            adjacencyMatrix = new LinkedHashMap<>();
        }
    }

    public CallGraphExtractor(String projectPath) {
        this.projectPath = projectPath;
        this.srcPath     = projectPath + "/src/main/java";
    }

    /**
     * Visite l'AST JavaParser pour extraire les appels de méthodes
     * et les injections @Autowired.
     */
    private class CallVisitor extends VoidVisitorAdapter<CallGraphResult> {

        private String currentClass   = "Unknown";
        private String currentPackage = "unknown";

        // ── Entrée dans une classe ──────────────────────────────
        @Override
        public void visit(ClassOrInterfaceDeclaration cls, CallGraphResult result) {
            currentClass = cls.getNameAsString();
            super.visit(cls, result);
        }

        // ── Appel de méthode : obj.method() ────────────────────
        @Override
        public void visit(MethodCallExpr call, CallGraphResult result) {

            String toClass   = call.getScope()
                    .map(Object::toString)
                    .orElse("this");
            String method    = call.getNameAsString();
            int    line      = call.getBegin()
                    .map(p -> p.line)
                    .orElse(0);

            // Filtrer les appels triviaux (this, super, chaînes courtes)
            if (!toClass.equals("this") && toClass.length() > 1) {
                MethodCall mc = new MethodCall(
                        currentClass, currentPackage,
                        toClass, method,
                        "method_call", line
                );
                result.calls.add(mc);

                // Mettre à jour la matrice d'adjacence
                result.adjacencyMatrix
                        .computeIfAbsent(currentClass, k -> new LinkedHashSet<>())
                        .add(toClass);
            }

            super.visit(call, result);
        }

        // ── Champ @Autowired : injection de dépendance Spring ──
        @Override
        public void visit(FieldDeclaration field, CallGraphResult result) {

            boolean hasAutowired = field.getAnnotations().stream()
                    .map(AnnotationExpr::getNameAsString)
                    .anyMatch(a -> a.equals("Autowired") || a.equals("Inject"));

            if (hasAutowired) {
                String depType = field.getVariables().get(0)
                        .getTypeAsString();
                String depName = field.getVariables().get(0)
                        .getNameAsString();
                int line = field.getBegin()
                        .map(p -> p.line)
                        .orElse(0);

                MethodCall mc = new MethodCall(
                        currentClass, currentPackage,
                        depType, depName,
                        "autowired", line
                );
                result.calls.add(mc);

                result.adjacencyMatrix
                        .computeIfAbsent(currentClass, k -> new LinkedHashSet<>())
                        .add(depType);
            }

            super.visit(field, result);
        }
    }

    /**
     * Point d'entrée : parcourt tous les .java et extrait le call graph.
     */
    public void extract() throws IOException {

        CallGraphResult result = new CallGraphResult();
        CallVisitor     visitor = new CallVisitor();

        // Chercher tous les fichiers .java
        List<Path> javaFiles = Files.walk(Paths.get(srcPath))
                .filter(p -> p.toString().endsWith(".java"))
                .collect(Collectors.toList());

        System.out.println("🔍 " + javaFiles.size() + " fichiers .java trouvés");

        int parsed = 0;
        int errors = 0;

        for (Path file : javaFiles) {
            try {
                // ── JavaParser parse le fichier en AST ──────────
                CompilationUnit cu = StaticJavaParser.parse(file);

                // Récupérer le package
                String pkg = cu.getPackageDeclaration()
                        .map(pd -> pd.getNameAsString())
                        .orElse("default");

                // Mettre à jour le package dans le visitor
                visitor.currentPackage = pkg;

                // Visiter l'AST
                visitor.visit(cu, result);
                parsed++;

                System.out.println("  ✅ " + file.getFileName() + " → parsé");

            } catch (Exception e) {
                errors++;
                System.out.println("  ⚠️  " + file.getFileName() + " → erreur: " + e.getMessage());
            }
        }

        // ── Statistiques ────────────────────────────────────────
        Set<String> classes = result.calls.stream()
                .map(c -> c.fromClass)
                .collect(Collectors.toSet());

        result.summary.put("totalFiles",    javaFiles.size());
        result.summary.put("parsedFiles",   parsed);
        result.summary.put("errorFiles",    errors);
        result.summary.put("totalClasses",  classes.size());
        result.summary.put("totalCalls",    result.calls.size());
        result.summary.put("classes",       new ArrayList<>(classes));

        System.out.println("\n📊 Statistiques :");
        System.out.println("   Classes analysées : " + classes.size());
        System.out.println("   Appels extraits   : " + result.calls.size());
        System.out.println("   Fichiers parsés   : " + parsed + "/" + javaFiles.size());

        System.out.println("\n🔍 Aperçu des 5 premiers appels :");
        result.calls.stream().limit(5).forEach(c ->
                System.out.printf("   %s → %s.%s() [ligne %d]%n",
                        c.fromClass, c.toClass, c.methodName, c.lineNumber));

        // ── Sauvegarder en JSON ─────────────────────────────────
        saveJson(result, outputDir + "/call_graph.json");
    }

    private void saveJson(Object obj, String path) throws IOException {
        new File(outputDir).mkdirs();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (Writer w = new FileWriter(path)) {
            gson.toJson(obj, w);
        }
        System.out.println("\n✅ Call graph sauvegardé → " + path);
    }
}
