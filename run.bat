@echo off
REM ============================================================
REM  run.bat — Compiler et lancer l'extraction (Windows)
REM ============================================================
REM  Usage : run.bat <chemin-vers-spring-petclinic>
REM  Exemple : run.bat C:\Users\hp\Downloads\spring-petclinic
REM ============================================================

echo ============================================
echo   JavaParser Feature Extractor
echo ============================================

REM Vérifier Maven
mvn -version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Maven non trouvé. Installe-le depuis https://maven.apache.org
    pause
    exit /b 1
)

REM Vérifier Java
java -version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Java non trouvé. Installe-le depuis https://adoptium.net
    pause
    exit /b 1
)

echo [1/2] Compilation du projet...
call mvn clean package -q

IF %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Compilation échouée.
    pause
    exit /b 1
)

echo [2/2] Lancement de l'extraction...

REM Chemin vers spring-petclinic (argument ou défaut)
SET PROJECT_PATH=%1
IF "%PROJECT_PATH%"=="" SET PROJECT_PATH=..\spring-petclinic

echo     Projet analysé : %PROJECT_PATH%
echo.

java -jar target\extractor-all.jar "%PROJECT_PATH%"

echo.
echo Terminé ! Voir les fichiers dans outputs/
pause
