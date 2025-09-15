@echo off
REM RSS Server Service Wrapper for Windows
REM Ensures the RTX 5090 RSS backend runs continuously

echo [%date% %time%] RSS Server Service Starting...

:START_SERVER
echo [%date% %time%] Starting RSS backend...

cd C:\Users\zhong\Dropbox\Workspace\Hackathon\video-rss-aggregator
set PYTHONIOENCODING=utf-8

REM Start the server
python rtx5090_incremental_backend.py

REM If we reach here, the server exited
echo [%date% %time%] Server exited unexpectedly, restarting in 10 seconds...
timeout /t 10 /nobreak

goto START_SERVER