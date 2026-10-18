@echo off
setlocal
if exist "%~dp0distilbert-thematic-analysis\app.py" (
    cd /d "%~dp0distilbert-thematic-analysis"
) else (
    cd /d "%~dp0"
)
if not defined THEMATIC_OPEN_BROWSER set "THEMATIC_OPEN_BROWSER=1"
where py >nul 2>&1
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.14 and the packages in requirements.txt.
    pause
    exit /b 1
)
py -3.14 -c "import fastapi, uvicorn, onnxruntime, transformers, pandas, sklearn, openpyxl, multipart" >nul 2>&1
if errorlevel 1 (
    echo Required packages are missing from Python 3.14.
    echo Run: py -3.14 -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo Starting Thematic Analysis Studio at http://localhost:8050
echo Keep this window open while using the app. Press Ctrl+C to stop.
py -3.14 app.py
if errorlevel 1 pause
