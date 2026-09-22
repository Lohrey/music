@echo off
chcp 65001 >nul
title Musik machen - Update
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
echo  Neuste Version wird geholt (vorher "Musik machen" schliessen) ...
uv run --no-project --python 3.11 python tools\installer.py update %*
pause
