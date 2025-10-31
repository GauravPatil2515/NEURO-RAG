@echo off
echo.
echo ========================================
echo   PUSH TO HUGGING FACE SPACES
echo ========================================
echo.
echo When prompted for credentials:
echo   Username: GauravPatil2515
echo   Password: [YOUR HF ACCESS TOKEN]
echo.
echo Get token from: https://huggingface.co/settings/tokens
echo.
pause
echo.
echo Pushing to Hugging Face...
echo.
git push huggingface main
echo.
if %errorlevel% == 0 (
    echo ========================================
    echo   SUCCESS! Check your Space at:
    echo   https://huggingface.co/spaces/GauravPatil2515/neuro-rag
    echo ========================================
) else (
    echo ========================================
    echo   FAILED! Common issues:
    echo   1. Wrong password - use ACCESS TOKEN
    echo   2. Token needs 'write' permission
    echo   3. Space doesn't exist yet
    echo ========================================
)
echo.
pause
