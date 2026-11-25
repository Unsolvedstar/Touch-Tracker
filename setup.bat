@echo off
rem One-time install for Windows. Needs Python 3.10 or newer from python.org ("Add python.exe to PATH" ticked).
cd /d "%~dp0"
python --version >nul 2>&1 || (echo Python was not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH". & pause & exit /b 1)
python -c "import sys; sys.exit(sys.version_info < (3, 10))" || (echo Python 3.10 or newer is needed. & pause & exit /b 1)
if not exist .venv python -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
echo Installing. This downloads several hundred MB, so use Wi-Fi if you can.
.venv\Scripts\python.exe -m pip install --default-timeout=120 --retries 10 -r requirements.txt || (echo Install failed. Check your internet connection and run setup.bat again. & pause & exit /b 1)
echo Downloading the detection and number-reading models...
.venv\Scripts\python.exe fetch_models.py || (echo Model download failed. Run setup.bat again when online. & pause & exit /b 1)
if not exist videos mkdir videos
echo.
echo Done. Copy a match video into the "videos" folder, then double-click run.bat.
pause
