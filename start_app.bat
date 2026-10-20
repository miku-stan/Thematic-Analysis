@echo off
setlocal
if exist "%~dp0distilbert-thematic-analysis\app.py" (
    cd /d "%~dp0distilbert-thematic-analysis"
) else (
    cd /d "%~dp0"
)
if not defined THEMATIC_OPEN_BROWSER set "THEMATIC_OPEN_BROWSER=1"

:: Resolve python executable
set "PY_CMD="

:: Check virtual environments first
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_CMD="%~dp0venv\Scripts\python.exe""
    goto :python_detected
)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD="%~dp0.venv\Scripts\python.exe""
    goto :python_detected
)

:: Try detected Python environments
for %%C in ("py -3.12" "py -3.14" "py -3" "py" "python") do (
    %%~C -c "import sys; sys.exit(0)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=%%~C"
        goto :python_detected
    )
)

:python_detected
if "%PY_CMD%"=="" (
    echo [ERROR] No suitable Python runtime was found.
    echo Please install Python 3.12 or 3.10+ and ensure it is added to PATH.
    pause
    exit /b 1
)

:: Check required dependencies
%PY_CMD% -c "import fastapi, uvicorn, onnxruntime, transformers, pandas, sklearn, openpyxl, multipart" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Required packages are missing. Installing dependencies from requirements.txt...
    %PY_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        echo Please run manually: %PY_CMD% -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo =======================================================
echo Starting Thematic Analysis Studio at http://localhost:8050
echo Using Python runtime: %PY_CMD%
echo Keep this window open while using the app. Press Ctrl+C to stop.
echo =======================================================
%PY_CMD% app.py
if errorlevel 1 pause

