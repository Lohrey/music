@echo off
chcp 65001 >nul
title Musik machen - Songs zurueckholen
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
rem Eine Backup-ZIP auf diese Datei ziehen - oder einfach doppelklicken (nimmt das neuste Backup).
uv run --no-project --python 3.11 python tools\installer.py restore %1
pause
