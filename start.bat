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

:: 2. Run the FastAPI Backend
echo [*] Starting FastAPI Backend...
start cmd /k "python -m src.api"

:: 3. Run the Vite Frontend
echo [*] Starting React Frontend...
cd frontend
start cmd /k "npm run dev"

echo.
echo ==========================================
echo       MeGPT is now running!
echo       Frontend: http://localhost:5173
echo       Backend:  http://localhost:8000
echo ==========================================
pause
