@echo off
REM Start Academicon Web UI with OpenRouter (Fast!)

echo ================================================
echo Academicon Code Assistant (OpenRouter)
echo ================================================
echo.
echo Starting Web UI on http://127.0.0.1:7860
echo Using model from .env: %MODEL_NAME%
echo.
echo Press Ctrl+C to stop
echo ================================================
echo.

D:\LOCAL-CODER\academicon-agent-env\Scripts\python.exe web_ui_openrouter.py

pause
