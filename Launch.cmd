@echo off
cd /d "%~dp0"
python -X utf8 -B solution.py gui
if errorlevel 1 pause
