# NeuroRAG Flask Server Startup Script
# PowerShell version for advanced users

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NeuroRAG Flask Server Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set environment variables to avoid TensorFlow issues
$env:USE_TF = '0'
$env:USE_TORCH = '1'
$env:TRANSFORMERS_NO_TF = '1'
$env:TF_ENABLE_ONEDNN_OPTS = '0'
$env:PYTHONUNBUFFERED = '1'

# Change to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Starting Flask server on http://127.0.0.1:5000" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: Keep this window open!" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "Open your browser to: " -NoNewline
Write-Host "http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run Flask app
& "C:\Users\GAURAV PATIL\AppData\Local\Programs\Python\Python312\python.exe" app_flask.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Red
Write-Host "Flask server stopped." -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Red
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
