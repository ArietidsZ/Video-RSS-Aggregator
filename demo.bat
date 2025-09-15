@echo off
REM Video RSS Aggregator - Windows One-Click Demo Script
REM Competition Ready Demo for Fresh Windows Machine

echo ========================================
echo  Video RSS Aggregator - Competition Demo
echo  Windows Edition
echo ========================================
echo.

REM Check if Python is installed
echo Checking Requirements...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo.
    echo Please install Node.js from: https://nodejs.org/
    pause
    exit /b 1
)

echo Requirements check passed!
echo.

REM Install Python dependencies
echo Installing Python dependencies...
pip install -q fastapi uvicorn aiohttp
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python packages
    echo Try running: pip install fastapi uvicorn aiohttp
    pause
    exit /b 1
)

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
call npm install --silent
cd ..

echo Dependencies installed!
echo.

REM Kill any existing processes on ports
echo Checking ports...
netstat -ano | findstr :8000 >nul 2>&1
if %errorlevel% equ 0 (
    echo Port 8000 is in use, attempting to free it...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
        taskkill /PID %%a /F >nul 2>&1
    )
)

netstat -ano | findstr :3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo Port 3000 is in use, attempting to free it...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000') do (
        taskkill /PID %%a /F >nul 2>&1
    )
)

REM Start backend server
echo Starting Services...
echo   Starting backend server (port 8000)...
start /B cmd /c "set PYTHONPATH=. && python -m uvicorn simple_backend:app --host 0.0.0.0 --port 8000 > backend.log 2>&1"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server
echo   Starting frontend server (port 3000)...
cd frontend
start /B cmd /c "npm start > ../frontend.log 2>&1"
cd ..

REM Wait for frontend to start
timeout /t 5 /nobreak >nul

echo Services started successfully!
echo.

REM Display status
echo ======================================
echo  Demo Status:
echo ======================================
echo   Frontend:  http://localhost:3000
echo   Backend:   http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo   RSS Feed:  http://localhost:8000/rss/bilibili
echo.

echo ======================================
echo  Demo Features:
echo ======================================
echo   - Real-time Chinese video platform data
echo   - AI-powered content summaries
echo   - RSS feed generation
echo   - Legal compliance framework
echo   - Multi-platform support
echo.

REM Check if services are running
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo Backend is running and healthy!
) else (
    echo WARNING: Backend health check failed
    echo Check backend.log for details
)

echo.
echo ======================================
echo  DEMO IS READY!
echo ======================================
echo.
echo Open http://localhost:3000 in your browser
echo.
echo To stop the demo:
echo   - Close this window
echo   - Or run: stop_demo.bat
echo.
echo Demo running in background...
echo Press any key to stop the demo
pause >nul

REM Cleanup when user presses a key
echo.
echo Stopping services...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
echo Demo stopped.