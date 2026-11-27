@echo off
title SPHURA (network)
cd /d "%~dp0"
rem Start the local AI if it is not running yet: the Ollama tray app if installed, else a hidden server
curl -s -m 2 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
  ) else (
    powershell -NoProfile -Command "Start-Process ollama -ArgumentList serve -WindowStyle Hidden"
  )
  timeout /t 5 /nobreak >nul
)
"%~dp0engine\.venv\Scripts\python.exe" -m sphura ui --lan
echo.
pause
