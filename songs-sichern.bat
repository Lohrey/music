@echo off
chcp 65001 >nul
title Musik machen - Songs sichern
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run --no-project --python 3.11 python tools\installer.py backup
pause
