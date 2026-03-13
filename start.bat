@echo off
echo ==========================================
echo       Launching MeGPT Personal Assistant
echo ==========================================
echo.

:: 1. Activate the conda environment
echo [*] Activating conda environment 'megpt'...
call conda activate megpt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not activate 'megpt' environment.
    echo Please make sure you have created it using:
    echo conda create -n megpt python=3.10
    pause
    exit /b
)

:: 2. Run the Streamlit application
echo [*] Starting Streamlit UI...
streamlit run frontend/app.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Streamlit failed to start.
    pause
)
