@echo off
chcp 65001 >nul
title Musik machen
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if not exist "app\env\Scripts\python.exe" (
  echo  Noch nicht installiert - starte zuerst die Installation ...
  call installieren.bat
)
uv run --no-project --python 3.11 python tools\installer.py start %*
pause
