@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run setup.bat first. & pause & exit /b 1)
echo Touch Tracker is starting. Your browser will open. Keep this window open while you use it.
.venv\Scripts\python.exe -m streamlit run app.py
pause
