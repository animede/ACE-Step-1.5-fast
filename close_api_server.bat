@echo off
rem Stop the API server listening on a port (default 8001):  close_api_server.bat [PORT]
setlocal
set PORT=%~1
if "%PORT%"=="" set PORT=8001
set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do (
    echo Stopping PID %%p
    taskkill /PID %%p /F >nul 2>nul && set FOUND=1
)
if "%FOUND%"=="0" echo No listener on port %PORT%.
endlocal
