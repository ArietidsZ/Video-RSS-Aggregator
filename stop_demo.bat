@echo off
REM Stop Demo Script for Windows

echo Stopping Video RSS Aggregator Demo...

REM Kill Python processes (backend)
taskkill /F /IM python.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo   Stopped backend server
)

REM Kill Node processes (frontend)
taskkill /F /IM node.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo   Stopped frontend server
)

REM Kill any uvicorn processes specifically
taskkill /F /FI "WINDOWTITLE eq uvicorn*" >nul 2>&1

REM Clean up log files
if exist backend.log del backend.log
if exist frontend.log del frontend.log

echo.
echo Demo stopped successfully!
echo All processes terminated and cleaned up

pause