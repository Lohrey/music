@echo off
chcp 65001 >nul
title Musik machen - Installation
cd /d "%~dp0"
echo.
echo  ============================================================
echo    Musik machen - Installation
echo    Beim ersten Mal 20-60 Minuten (ca. 30 GB Downloads).
echo    Einfach laufen lassen. Du kannst es jederzeit neu starten.
echo  ============================================================
echo.
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo  uv wird installiert ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>nul
if errorlevel 1 (
  echo.
  echo  [FEHLER] uv konnte nicht installiert werden. Internet pruefen und nochmal starten.
  pause
  exit /b 1
)
uv run --no-project --python 3.11 python tools\installer.py install %*
echo.
pause
