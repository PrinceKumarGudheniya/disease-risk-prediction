@echo off
echo ==========================================
echo  HealthPredict AI - Flask Server
echo ==========================================
cd /d "%~dp0"
echo Starting server at http://localhost:5000
echo Press Ctrl+C to stop.
echo.
python app\app.py
pause
