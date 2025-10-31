@echo off
title NeuroRAG - Mental Health AI Assistant
color 0A

echo.
echo ======================================================================
echo                 🧠 NeuroRAG - Mental Health AI Assistant
echo ======================================================================
echo.
echo   Starting Flask Server...
echo.

REM Set environment variables to avoid dependency issues
set USE_TF=0
set USE_TORCH=1
set TRANSFORMERS_NO_TF=1
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHONUNBUFFERED=1

REM Change to project directory (parent of scripts folder)
cd /d "%~dp0.."

echo   📍 Dashboard URL: http://127.0.0.1:5000
echo   📍 Alternative:   http://localhost:5000
echo.
echo   ⚠️  IMPORTANT: Keep this window OPEN while using the app!
echo   ⚠️  Press Ctrl+C to stop the server
echo.
echo ======================================================================
echo.

REM Start the Flask server
python run_server.py

echo.
echo ======================================================================
echo   Server stopped.
echo ======================================================================
echo.
pause
