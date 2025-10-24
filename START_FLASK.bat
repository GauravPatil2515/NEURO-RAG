@echo off
echo ========================================
echo   NeuroRAG Flask Server Startup
echo ========================================
echo.

REM Set environment variables to avoid TensorFlow issues
set USE_TF=0
set USE_TORCH=1
set TRANSFORMERS_NO_TF=1
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHONUNBUFFERED=1

REM Change to project directory
cd /d "%~dp0"

echo Starting Flask server on http://127.0.0.1:5000
echo.
echo IMPORTANT: Keep this window open!
echo Press Ctrl+C to stop the server
echo.
echo Open your browser to: http://127.0.0.1:5000
echo ========================================
echo.

REM Run Flask app
"C:\Users\GAURAV PATIL\AppData\Local\Programs\Python\Python312\python.exe" app_flask.py

echo.
echo ========================================
echo Flask server stopped.
echo ========================================
pause
