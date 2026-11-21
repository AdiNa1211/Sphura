@echo off
title SPHURA - After Stream
rem Double-click after you stop streaming/recording in OBS.
rem Optional: drag a video file onto this icon to process that file instead.
if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\after-stream.ps1"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\after-stream.ps1" -File "%~1"
)
echo.
pause
